import transformers
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
# from utils import _prepare_decoder_attention_mask
from torch.nn import CrossEntropyLoss
from gist_cllm.utils import make_gist_mask

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CllmTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        args = kwargs["args"]
        self.train_step_cnt = 0
        self.max_new_tokens = args.max_new_tokens
        self.use_gt_labels = args.use_gt_labels

    def training_step(self, model, inputs):
        self.train_step_cnt += 1
        return self.consistency_training_step(model, inputs)

    def consistency_training_step(self, model, inputs):

        max_new_tokens = self.max_new_tokens      

        jacobian_trajectory = inputs["jacobian_trajectory"]
        input_masks = inputs["attention_mask"]
        prompt_id_len = inputs["prompt_id_len"]
        bsz = jacobian_trajectory[0].shape[0]
        eos_reached = torch.tensor([False] * bsz).to(model.device)

        ### tokens generated after <eos> are set to <pad>
        for i in range(len(jacobian_trajectory)):
            for j in range(bsz):
                trajectory_len = torch.sum(input_masks, dim=-1)
                # find the first accurate <EOS>
                eos_positions = torch.where(jacobian_trajectory[i][j, :(trajectory_len[j]-max_new_tokens)]==self.tokenizer.eos_token_id)[0]
                if len(eos_positions)==0:
                    continue
                # otherwise, set tokens coming after the accurate <EOS> as pad 
                eos_reached[j] = True
                trajectory_copy = jacobian_trajectory[i].clone().detach()
                eos_pos = eos_positions[0]
                trajectory_copy[j, int(eos_pos)+1:] = self.tokenizer.pad_token_id
                jacobian_trajectory[i] = trajectory_copy  

        ## compute AutoRegression loss ###
        # use labels to avoid pattern collapse
        if self.use_gt_labels:
            inputs_labels = inputs['labels_ids']
        else:
            inputs_labels = inputs['teacher_output_ids']
        # TODO: check if it's right when batch size > 1
        labels = torch.tensor(inputs_labels).to(model.device)
        attention_mask = torch.full_like(labels, 1).to(model.device)
        label_student_model_output = model(labels, attention_mask)

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(model.device)
        attention_mask = jacobian_trajectory[-1] != self.tokenizer.pad_token_id
        logits_last =  self.get_logits(model, jacobian_trajectory[-1].clone().detach(), attention_mask)

        label_smoother = LabelSmoother(epsilon=0.1, ignore_index= -100)
        loss_ar = label_smoother(label_student_model_output, labels, shift_labels=True)
        loss_ar*=10
        if self.args.qlora:
            loss_ar.requires_grad = True
        print(f'loss ar: {loss_ar} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_ar)

        ### compute Consistency loss (global) ###
        # random select one point from trajectory
        i = random.choice(range(len(jacobian_trajectory))[:-1])

        # insert gist
        itr_index = int(inputs['jacobian_itr_id'][0].split('_')[-1])
        inputs_add_gist = jacobian_trajectory[i]
        label_add_gist = jacobian_trajectory[-1]
        gist_token_ids = []
        for j in range(max_new_tokens-self.args.max_new_tokens_unit, 0, -self.args.max_new_tokens_unit):
            GIST = f"<GIST{itr_index}>" if itr_index >= 10 else f"<GIST0{itr_index}>"
            assert len(self.tokenizer.encode(GIST)) == 2, f"GIST: {GIST}, encode: {self.tokenizer.encode(GIST)}"
            gist_token = self.tokenizer.encode(GIST)[-1]
            inputs_add_gist = torch.cat([inputs_add_gist[:, :-j], torch.full_like(inputs_add_gist, gist_token, device=inputs_add_gist.device)[:, :self.args.num_each_gist_token], inputs_add_gist[:, -j:]], dim=1)
            label_add_gist = torch.cat([label_add_gist[:, :-j], torch.full_like(label_add_gist, gist_token, device=label_add_gist.device)[:, :self.args.num_each_gist_token], label_add_gist[:, -j:]], dim=1)
            gist_token_ids.append(gist_token)
            itr_index += 1

        accepted_len = jacobian_trajectory[-1].shape[-1] - max_new_tokens
        attention_mask_gist = make_gist_mask(inputs_add_gist[:, accepted_len:], gist_token_ids, training=True, max_new_tokens=max_new_tokens, max_new_tokens_unit=self.args.max_new_tokens_unit)
        attention_mask_gist_full = torch.ones([1, 1, inputs_add_gist.shape[-1], inputs_add_gist.shape[-1]], device=attention_mask_gist.device)
        attention_mask_gist_full[:, :, -attention_mask_gist.shape[-1]:, -attention_mask_gist.shape[-1]:] = attention_mask_gist   

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
        attention_mask = inputs_add_gist != self.tokenizer.pad_token_id
        logits_i = self.get_logits(model, inputs_add_gist.clone().detach(), attention_mask, attention_mask_gist_full)


        # cross-entropy
        if self.args.use_cross_entropy:
            label_add_gist[:, :accepted_len] = -100
            shift_logits = logits_i[..., :-1, :].contiguous()
            shift_labels = label_add_gist[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss_consistency = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            ).float()
        else:
            output_mask = inputs_add_gist[..., 1:] == self.tokenizer.pad_token_id
            # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
            for j in range(bsz):
                end_of_mask_position = torch.where(inputs_add_gist[j, 1:] != label_add_gist[j, 1:])[0]
                if len(end_of_mask_position)==0:
                    output_mask[j, :] = True
                else:
                    output_mask[j, :end_of_mask_position[0]] = True
            
            gist_token_index = [torch.where(label_add_gist == i)[-1][0] for i in gist_token_ids]
            output_mask[:, gist_token_index] = True
            for i in range(len(gist_token_index)):
                if i == 0:
                    logits_add_gist = torch.cat((logits_last[:, :gist_token_index[i], :], logits_i[:, gist_token_index[i]:gist_token_index[i]+1, :]), dim=1)
                else:
                    logits_add_gist = torch.cat((logits_add_gist, logits_last[:, gist_token_index[i-1]+1:gist_token_index[i], :], logits_i[:, gist_token_index[i]:gist_token_index[i]+1, :]), dim=1)
            logits_add_gist = torch.cat((logits_add_gist, logits_last[:, -self.args.max_new_tokens_unit:, :]), dim=1)
            # logits_last_gist = self.get_logits(model, inputs_add_gist.clone().detach(), attention_mask, attention_mask_gist_full)
            loss_consistency = self.soft_cross_entropy(
                        logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                        logits_add_gist[..., :-1, :].to(logits_i.device).clone().detach().float(),
                        output_mask.to(logits_i.device)
            )
        if self.args.qlora:
            loss_consistency.requires_grad = True
        print(f'loss consistency {loss_consistency} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_consistency)
        
        # else:   
        if self.args.local_rank == 0:
            wandb.log({"ar loss": loss_ar})
            wandb.log({"consistency loss": loss_consistency})
            
        # sync processes
        torch.distributed.barrier()
        # total loss = ar_loss + consistency_global_loss
        loss = loss_ar.detach() + loss_consistency.detach()

        return loss
    

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(10)
        # Create custom DataLoader with shuffle set to False
        shuffle = True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "generator": generator
        }

        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    ###################### Helper Functions #############################
    def soft_cross_entropy(self, predicts, targets, padding_mask):
        # TODO: support batch_size >1 here.
        if (~padding_mask).sum() == 0:
            return 0*predicts[0][0][0]
        predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        entropy = -targets_prob * predict_log_prob
        expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
        entropy.masked_fill_(expand_mask, 0)
        mean_entropy = entropy.sum() / (~padding_mask).sum()
        return mean_entropy

    def get_logits(self, model, input_ids, attention_mask, attention_mask_gist_full=None):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_gist = attention_mask_gist_full
        ).logits
