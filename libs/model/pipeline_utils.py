from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.models.attention_processor import Attention
from diffusers.models import AutoencoderKL, UNet2DConditionModel
import torch
import numpy as np
import PIL

class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if self.is_pix2pix_zero and timestep is not None:
            # new bookkeeping to save the attention weights.
            if loss is None:
                self.reference_cross_attn_map[timestep.item()] = attention_probs.detach().cpu()
            # compute loss
            elif loss is not None:
                prev_attn_probs = self.reference_cross_attn_map.pop(timestep.item())
                loss.compute_loss(attention_probs, prev_attn_probs.to(attention_probs.device))

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def prepare_unet(unet: UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    pix2pix_zero_attn_procs = {}
    for name in unet.attn_processors.keys():
        module_name = name.replace(".processor", "")
        module = unet.get_submodule(module_name)
        if "attn2" in name:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True)
            module.requires_grad_(True)
        else:
            pix2pix_zero_attn_procs[name] = Pix2PixZeroAttnProcessor(is_pix2pix_zero=False)
            module.requires_grad_(False)

    unet.set_attn_processor(pix2pix_zero_attn_procs)
    return unet

def _in_step(config, step):
    in_step = False
    try:
        start_step = config.start_step
        end_step = config.end_step
        if start_step <= step < end_step:
            in_step = True
    except:
        in_step = False
    return in_step

def _classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block