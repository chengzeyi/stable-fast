from typing import Sequence
import torch

TensorLikeType = torch.Tensor


# This combines is_channels_last_strides_2d and is_channels_last_strides_3d in
# c10/core/MemoryFormat.h into one function
def are_strides_like_channels_last(
    shape: Sequence[int], strides: Sequence[int]
) -> bool:
    ndim = len(shape)

    if ndim == 4:
        # Check for channels_last_2d
        dim_order = [1, 3, 2, 0]
    elif ndim == 5:
        # Check for channels_last_3d
        dim_order = [1, 4, 3, 2, 0]
    else:
        return False

    if strides[1] == 0:
        return False

    min = 0
    for d in dim_order:
        if shape[d] == 0:
            return False
        if strides[d] < min:
            return False
        if d == 0 and min == strides[1]:
            return False
        min = strides[d]
        if strides[d] > 1:
            min *= shape[d]
    return True


def suggest_memory_format(x: TensorLikeType) -> torch.memory_format:
    if x.layout != torch.strided:
        return torch.contiguous_format

    if are_strides_like_channels_last(x.shape, x.stride()):
        return torch.channels_last if x.ndim == 4 else torch.channels_last_3d

    return torch.contiguous_format