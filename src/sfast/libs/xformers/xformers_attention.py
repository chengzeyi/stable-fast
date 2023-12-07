from typing import Optional
import torch
from xformers.ops import (memory_efficient_attention, AttentionOp)
from xformers import ops
from sfast.utils.custom_python_operator import register_custom_python_operator

OP_STR_MAP = {
    ops.MemoryEfficientAttentionCutlassFwdFlashBwOp:
    'MemoryEfficientAttentionCutlassFwdFlashBwOp',
    ops.MemoryEfficientAttentionCutlassOp: 'MemoryEfficientAttentionCutlassOp',
    ops.MemoryEfficientAttentionFlashAttentionOp:
    'MemoryEfficientAttentionFlashAttentionOp',
    ops.MemoryEfficientAttentionOp: 'MemoryEfficientAttentionOp',
    ops.MemoryEfficientAttentionTritonFwdFlashBwOp:
    'MemoryEfficientAttentionTritonFwdFlashBwOp',
    ops.TritonFlashAttentionOp: 'TritonFlashAttentionOp',
}

STR_OP_MAP = {v: k for k, v in OP_STR_MAP.items()}


def xformers_memory_efficient_attention_torch_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        op: Optional[str] = None):
    if op is not None:
        op = STR_OP_MAP[op]
    hidden_states = memory_efficient_attention(query,
                                               key,
                                               value,
                                               attn_bias,
                                               p,
                                               scale,
                                               op=op)
    return hidden_states


register_custom_python_operator(
    'sfast_xformers::memory_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, float p, float? scale, str? op) -> Tensor',
    xformers_memory_efficient_attention_torch_op)


def xformers_memory_efficient_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        *,
        op: Optional[AttentionOp] = None):
    if op is not None:
        op = OP_STR_MAP[op]
    return torch.ops.sfast_xformers.memory_efficient_attention(
        query, key, value, attn_bias, p, scale, op)
