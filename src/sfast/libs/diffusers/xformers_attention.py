import torch
from diffusers.models.attention_processor import Attention
from sfast.utils.patch import patch_module


def patch_all_attention_modules(m):
    return patch_module(m, lambda stack: isinstance(stack[-1][1], Attention),
                        patch_attention_module)


def patch_attention_module(m):
    assert isinstance(m, Attention)

    m.batch_to_head_dim = batch_to_head_dim.__get__(m)
    m.head_to_batch_dim = head_to_batch_dim.__get__(m)

    return m


def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
    r"""
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
    is the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    # head_size = self.heads
    # batch_size, seq_len, dim = tensor.shape
    # tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
    # tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
    # return tensor

    batch_size, seq_len, head_size, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size * dim)
    return tensor


def head_to_batch_dim(self,
                      tensor: torch.Tensor,
                      out_dim: int = 3) -> torch.Tensor:
    r"""
    Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
    the number of heads initialized while constructing the `Attention` class.

    Args:
        tensor (`torch.Tensor`): The tensor to reshape.
        out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
            reshaped to `[batch_size * heads, seq_len, dim // heads]`.

    Returns:
        `torch.Tensor`: The reshaped tensor.
    """
    # head_size = self.heads
    # batch_size, seq_len, dim = tensor.shape
    # tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    # tensor = tensor.permute(0, 2, 1, 3)

    # if out_dim == 3:
    #     tensor = tensor.reshape(batch_size * head_size, seq_len, dim // head_size)

    # return tensor
    head_size = self.heads
    batch_size, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
    return tensor
