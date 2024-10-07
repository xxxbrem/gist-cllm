from dataclasses import dataclass, field
import json
import math
import pathlib
import functools
from typing import Dict, Optional, Sequence, List, Tuple
import random
from tqdm import tqdm
import torch.nn.functional as F
import sqlite3
import time
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

import torch.nn.functional as F
from transformers import LlamaModel,LlamaForCausalLM
import argparse
from gist_cllm.utils import _prepare_decoder_attention_mask, make_gist_mask

def delete_false_key_value(
        self,
        num_of_false_tokens,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
   
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][..., :-num_of_false_tokens, :]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][..., :-num_of_false_tokens, :]

@torch.inference_mode()
def jacobi_forward(
    self,
    input_ids: torch.LongTensor = None,
    tokenizer=None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    chat: Optional[bool] = False,
):
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001,  dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0

        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0

        prev_len = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.001, dim=-1)

            next_point = torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)

            first_false_index = torch.where(torch.eq(current_point[0], next_point[0]) == False)[0]
            
            jacobian_trajectory.append(next_point)

            if len(first_false_index) > 0:
                fast_forward_cnt = first_false_index[0].item()

                past_key_values.delete_false_key_value(seq_length - fast_forward_cnt) # delete the false keys & values
            else:
                fast_forward_cnt = torch.sum(torch.eq(current_point, next_point)).item()

                accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]         
                first_correct_token = all_shift_one_token[:,-1]   
                if chat:
                    if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length + fast_forward_cnt]:
                        eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                        eos_position = eos_positions[0]
                        generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                    else:
                        generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length + fast_forward_cnt], skip_special_tokens=True)

                    print(generated_str[prev_len:], flush=True, end="")
                    prev_len = len(generated_str)
                break 

            accurate_n_gram[0, accurate_length : accurate_length + fast_forward_cnt] = next_point[0, :fast_forward_cnt]
            accurate_length += fast_forward_cnt
            next_point = next_point[0, fast_forward_cnt:].view(1,-1) # only false tokens should be re-generated

            if chat:
                if tokenizer.eos_token_id in accurate_n_gram[0, :accurate_length]:
                    eos_positions = torch.where(accurate_n_gram[0]==tokenizer.eos_token_id)[0]
                    eos_position = eos_positions[0]

                    generated_str = tokenizer.decode(accurate_n_gram[0, :eos_position], skip_special_tokens=True)
                else:
                    generated_str = tokenizer.decode(accurate_n_gram[0, :accurate_length], skip_special_tokens=True)

                print(generated_str[prev_len:], flush=True, end="")
                prev_len = len(generated_str)

            iter_counter += 1

        return accurate_n_gram, first_correct_token, iter_counter, accurate_length


@torch.inference_mode()
def jacobi_forward_profiling(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        next_point = input_ids
        jacobian_trajectory.append(next_point)

        iter_counter = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            jacobian_trajectory.append(next_point)
            
            if torch.all(torch.eq(current_point, next_point)).item():    
                #print('Successfully break!')
                #print(next_point)
                first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter
    
@torch.inference_mode()
def gist_jacobi_forward(
    self,
    tokenizer,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    max_new_tokens: Optional[int] = None,
    prefill_phase: Optional[bool] = False,
    max_new_tokens_unit: Optional[int] = None,
    num_gist_tokens: Optional[int] = None,
    gist_token_id: Optional[int] = None
):
    
    assert use_cache == True

    if input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    
    if prefill_phase: # prefill phase, just compute the keys & values of prompt
        # self.model is the instance of class LlamaModel
        inputs_embeds = self.model.embed_tokens(input_ids)
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length) 

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa :
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        for decoder_layer in self.model.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[1]

        hidden_states = self.model.norm(hidden_states)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        predict_next_tokens = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)
        first_correct_token = predict_next_tokens[:, -1]
        return next_decoder_cache, first_correct_token
    else: # generation phase, input as random_initilized point and output as fixed point
        
        jacobian_trajectory = []
        accurate_n_gram = torch.zeros_like(input_ids).to(input_ids.device)
        accurate_length = 0
        # next_point = input_ids
        jacobian_trajectory.append(input_ids)
        if num_gist_tokens > 0:
            gist_token_id = tokenizer.encode('<GIST>')[-1]
            # insert gist tokens
            gist_tokens = torch.tensor(gist_token_id, device=input_ids.device).repeat(num_gist_tokens).unsqueeze(0)
            input_ids_new = []
            gist_index = []
            index_count = 0
            for i in range(0, input_ids.shape[-1]-max_new_tokens_unit, max_new_tokens_unit):
                input_ids_new.append(input_ids[:, i:i+max_new_tokens_unit])
                index_count += max_new_tokens_unit
                input_ids_new.append(gist_tokens)
                gist_index += [j for j in range(index_count, index_count+num_gist_tokens)]
                index_count += num_gist_tokens
            input_ids_new.append(input_ids[:, -max_new_tokens_unit:])
            next_point = torch.cat(input_ids_new, dim=1)
            # gist_index = torch.where(torch.cat(input_ids_new, dim=1) == gist_token_id)[-1]
            
            # i = max_new_tokens_unit - 1
            # while i < input_ids.shape[-1]:
            #     tmp = num_gist_tokens
            #     while tmp:
            #         gist_index.append(i)
            #         i += 1
            #         tmp -= 1
            #     i += max_new_tokens_unit
        else:
            next_point = input_ids

        iter_counter = 0
        every_unit_count = []
        complete_unit = 0
        while True:

            current_point = next_point
            inputs_embeds = self.model.embed_tokens(current_point)
            attention_mask = None
            position_ids = None
            seq_length = current_point.shape[1]
            if use_cache:
                use_legacy_cache = not isinstance(past_key_values, Cache)
                if use_legacy_cache:
                    past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_length = past_key_values.get_usable_length(seq_length) 
            # print(past_key_values_length) # return previous_seq_length
            if position_ids is None:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                position_ids = torch.arange(
                    past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
                )
                position_ids = position_ids.unsqueeze(0)

            if num_gist_tokens > 0:
                seq_length_with_past = seq_length
                seq_length_with_past = seq_length_with_past + past_key_values_length
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
                    past_key_values_length,
                )

                attention_mask_gist_float = torch.full_like(
                    attention_mask, torch.tensor(torch.finfo(attention_mask.dtype).min)
                )
                attention_mask_gist = make_gist_mask(current_point, gist_token_id, past_key_values_length)
                if attention_mask_gist is not None:
                    attention_mask_gist_float = attention_mask_gist_float.masked_fill(
                        attention_mask_gist.bool(), 0.0
                    )
                    attention_mask = attention_mask + attention_mask_gist_float
            elif self.model._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            elif self.model._use_sdpa :
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # 4d mask is passed through the layers
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
            # embed positions
            hidden_states = inputs_embeds

            # decoder layers            
            for decoder_layer in self.model.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                )

                hidden_states = layer_outputs[0]

            hidden_states = self.model.norm(hidden_states)

            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)

            logits = logits.float()
            all_shift_one_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1) / 0.01, dim=-1)
            next_point= torch.cat((current_point[0, 0].view(1,-1), all_shift_one_token[0, :seq_length-1].view(1,-1)), dim=-1)
            if num_gist_tokens > 0:
                # force gist token
                next_point[:, gist_index] = gist_token_id
                # remove gist token
                mask = torch.ones_like(next_point, dtype=torch.bool, device=next_point.device)
                if len(gist_index) > 0:
                    mask[:, gist_index] = False
                jacobian_trajectory.append(next_point[mask].unsqueeze(0))
                # check unit
                if len(gist_index) > 0:
                    if torch.all(torch.eq(current_point[:, :gist_index[0]], next_point[:, :gist_index[0]])).item():
                        every_unit_count.append(iter_counter)
                        next_point = torch.cat((next_point[:, :gist_index[0]], next_point[:, gist_index[1]+1:]), dim=1)
                        gist_index = gist_index[num_gist_tokens:]
                        complete_unit += 1
                else:
                    if torch.all(torch.eq(current_point[:, -max_new_tokens_unit:], next_point[:, -max_new_tokens_unit:])).item():
                        every_unit_count.append(iter_counter)
                        first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                        break
            else: 
                jacobian_trajectory.append(next_point)
                
                if torch.all(torch.eq(current_point, next_point)).item():    
                    #print('Successfully break!')
                    #print(next_point)
                    first_correct_token = torch.argmax(torch.nn.functional.softmax(logits, dim=-1), dim=-1)[:,-1]
                    break
            past_key_values.delete_false_key_value(seq_length)

            iter_counter += 1

        return jacobian_trajectory[:-1], next_point, first_correct_token, iter_counter