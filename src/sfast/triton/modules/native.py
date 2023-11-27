import logging
# import functools
# import torch
import torch.nn as nn
from torch._prims_common import suggest_memory_format
from .. import torch_ops as TTO

logger = logging.getLogger()

# try:
#     import xformers
#     from xformers.triton.fused_linear_layer import _fused_linear_triton
# except ImportError:
#     logger.warning(
#         'xformers not found, some Triton optimizations will be disabled')
#     xformers = None


class TritonConv2D(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)


class TritonLinear(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        weight = self.module.weight
        x = TTO.contiguous(x, memory_format=suggest_memory_format(weight))
        return self.module(x, *args, **kwargs)
        # if xformers is None:
        #     return self.module(x, *args, **kwargs)
        # else:
        #     grad_mode = torch.is_grad_enabled()
        #     bias = self.module.bias
        #     return _fused_linear_triton.apply(
        #         x, weight, bias, 0, grad_mode and weight.requires_grad,
        #         False if bias is None else grad_mode and bias.requires_grad,
        #         False)


class TritonGroupNorm(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        # x = TTO.contiguous(x)
        # return self.module(x, *args, **kwargs)
        module = self.module
        # x = TTO.contiguous(x)
        return TTO.group_norm(x, module.num_groups, module.weight, module.bias,
                              module.eps)


class TritonGroupNormSiLU(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.training = module.training

    def forward(self, x, *args, **kwargs):
        # x = TTO.contiguous(x)
        # return self.module(x, *args, **kwargs)
        module = self.module
        # x = TTO.contiguous(x)
        return TTO.group_norm_silu(x, module.num_groups, module.weight,
                                   module.bias, module.eps)
