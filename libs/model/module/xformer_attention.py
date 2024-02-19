import math
from typing import Optional, Callable

import xformers
from diffusers.models.attention_processor import Attention


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


class MySelfAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 scale: float = 1.0, ):

        residual = hidden_states

        args = (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        self.attention_mask = attention_mask
        self.attn = attn
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, scale=scale)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, scale=scale)
        value = attn.to_v(encoder_hidden_states, scale=scale)

        # Record the Q,K,V for PCA guidance
        self.key = key
        self.query = query
        self.value = value
        self.hidden_state = hidden_states.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def prep_unet_attention(unet):
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            module.set_processor(MySelfAttnProcessor())
    return unet


def get_self_attn_feat(unet, injection_config):
    hidden_state_dict = dict()
    query_dict = dict()
    key_dict = dict()
    value_dict = dict()
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and 'attn1' in name and classify_blocks(injection_config.blocks, name=name):
            res = int(math.sqrt(module.processor.hidden_state.shape[1]))
            bs = module.processor.hidden_state.shape[0]
            hidden_state_dict[name] = module.processor.hidden_state.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            res = int(math.sqrt(module.processor.query.shape[1]))
            query_dict[name] = module.processor.query.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            key_dict[name] = module.processor.key.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            value_dict[name] = module.processor.value.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
    return hidden_state_dict, query_dict, key_dict, value_dict


def clean_attn_buffer(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and 'attn' in name:
            if 'injection_config' in module.processor.__dict__.keys():
                module.processor.injection_config = None
            if 'injection_mask' in module.processor.__dict__.keys():
                module.processor.injection_mask = None
            if 'obj_index' in module.processor.__dict__.keys():
                module.processor.obj_index = None
            if 'pca_weight' in module.processor.__dict__.keys():
                module.processor.pca_weight = None
            if 'pca_weight_changed' in module.processor.__dict__.keys():
                module.processor.pca_weight_changed = None
            if 'pca_info' in module.processor.__dict__.keys():
                module.processor.pca_info = None
            if 'step' in module.processor.__dict__.keys():
                module.processor.step = None
