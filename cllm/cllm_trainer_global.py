import transformers
import torch
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import wandb
import random
from torch.utils.data import DataLoader
from utils import _prepare_decoder_attention_mask
from torch.nn import CrossEntropyLoss

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

        ### compute AutoRegression loss ###
        # use labels to avoid pattern collapse
        if self.use_gt_labels:
            labels = inputs['labels_ids']
        else:
            labels = inputs['teacher_output_ids']
        # TODO: check if it's right when batch size > 1
        labels = torch.tensor(labels).to(model.device)
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

        attention_mask = torch.full_like(jacobian_trajectory[0], 1).to(jacobian_trajectory[0].device)
        attention_mask = jacobian_trajectory[i] != self.tokenizer.pad_token_id
        logits_i = self.get_logits(model, jacobian_trajectory[i].clone().detach(), attention_mask)

        output_mask = jacobian_trajectory[i][..., 1:] == self.tokenizer.pad_token_id
        # We do not calculate the cross entrophy of same logits to alleviate misleading gradients
        for j in range(bsz):
            end_of_mask_position = torch.where(jacobian_trajectory[i][j, 1:] != jacobian_trajectory[-1][j, 1:])[0]
            if len(end_of_mask_position)==0:
                output_mask[j, :] = True
            else:
                output_mask[j, :end_of_mask_position[0]] = True
        
        loss_global = self.soft_cross_entropy(
                    logits_i[..., :-1, :].float(), # logits generated by the last token is dropped
                    logits_last[..., :-1, :].to(logits_i.device).clone().detach().float(),
                    output_mask.to(logits_i.device)
        )
        if self.args.qlora:
            loss_global.requires_grad = True
        print(f'loss global {loss_global} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_global)
        
        ### compute Gist loss ###
        # line_break_id = self.tokenizer.encode("\n")[-1]
        # batch size = 1
        attention_mask = None
        batch_size = 0
        right_answer = jacobian_trajectory[-1]
        loss_gist = None
        if self.args.num_gist_token > 0:
            gist_token = self.args.gist_token
            attention_mask_gist = self.args.attention_mask_gist
            
            trajectory_decode = self.tokenizer.decode(right_answer[batch_size])
            line_break_id = self.tokenizer.encode("\n")[-1]
            _, line_break_id_index = torch.where(right_answer == line_break_id)
            assert line_break_id_index[1] + 3 == line_break_id_index[2]
            
            start = line_break_id_index[2]
            seq_length = len(right_answer[batch_size])
            for i in range(start, seq_length - max_new_tokens, max_new_tokens):
                # insert gist tokens
                adjacent_seq = torch.cat((right_answer[:, start:start+max_new_tokens], \
                    torch.full_like(right_answer, gist_token, device=right_answer.device)[:, :self.args.num_gist_token], \
                    right_answer[:, start+max_new_tokens:start+2*max_new_tokens] \
                    if start+2*max_new_tokens <= seq_length \
                    else torch.nn.functional.pad(right_answer[:, start+max_new_tokens:], (0, max_new_tokens+start+2*max_new_tokens-seq_length), value=self.tokenizer.pad_token_id) \
                    ), dim=1)
                # predict
                input_ids = torch.cat((right_answer[:, :start], adjacent_seq), dim=1)
                inputs_embeds =model.get_input_embeddings()(input_ids)
                #   past_key_values_length ?
                attention_mask = self.get_attention_mask(attention_mask, inputs_embeds, attention_mask_gist, 0)
                logits_i = self.get_logits(model, input_ids.clone().detach(), attention_mask)
                # loss
                input_ids[:, :start] = -100
                # Shift so that tokens < n predict n
                shift_logits = logits_i[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
                )
                loss_gist += loss.detach
        
        if self.args.qlora:
            loss_gist.requires_grad = True
        print(f'loss gist: {loss_gist} computed! performing backward pass...')
        with self.accelerator.accumulate(model):
            self.accelerator.backward(loss_gist)
        
        
        if self.args.local_rank == 0:
            wandb.log({"ar loss": loss_ar})
            wandb.log({"consistency loss": loss_global})
            wandb.log({"gist loss": loss_gist})

            
        # sync processes
        # torch.distributed.barrier()
        # total loss = ar_loss + consistency_global_loss
        loss = loss_ar.detach() + loss_global.detach() + loss_gist.detach

        return loss
    

    def log(self, logs):
        # Remove the 'loss' entry with value 0 before calling the superclass method
        if 'loss' in logs and logs['loss'] == -1:
            del logs['loss']

        # Call the original `log` method of the `Trainer` class
        super().log(logs)

    def get_train_dataloader(self):
        # Create custom DataLoader with shuffle set to False
        shuffle = True
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "shuffle": shuffle,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            # "collate_fn": lambda x: x
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

    def get_logits(self, model, input_ids, attention_mask):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

    def get_attention_mask(self, attention_mask, inputs_embeds, attention_mask_gist, past_key_values_length):
        batch_size, seq_length, _ = inputs_embeds.shape
        seq_length_with_past = seq_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = _prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length
        )

        attention_mask_gist_float = torch.full_like(
            attention_mask, torch.tensor(torch.finfo(attention_mask.dtype).min)
        )
        attention_mask_gist_float = attention_mask_gist_float.masked_fill(
            attention_mask_gist.bool(), 0.0
        )
        return attention_mask + attention_mask_gist_float