from sfast.utils.patch import patch_module


def patch_conv2d(m):
    from torch.nn import Conv2d
    from .native import TritonConv2D
    return patch_module(m, lambda stack: isinstance(stack[-1][1], Conv2d),
                        TritonConv2D)


def patch_linear(m):
    from torch.nn import Linear
    from .native import TritonLinear
    return patch_module(m, lambda stack: isinstance(stack[-1][1], Linear),
                        TritonLinear)


def patch_group_norm(m):
    from torch.nn import GroupNorm
    from .native import TritonGroupNorm
    return patch_module(m, lambda stack: isinstance(stack[-1][1], GroupNorm),
                        TritonGroupNorm)


def patch_group_norm_silu(m):
    from torch.nn import (Sequential, GroupNorm, SiLU)
    from .native import TritonGroupNormSiLU

    def filter_func(stack):
        seq = stack[-1][1]
        if not isinstance(seq, Sequential):
            return False
        return len(seq) >= 2 and isinstance(seq[0], GroupNorm) and isinstance(
            seq[1], SiLU)

    def patch_func(module):
        return Sequential(TritonGroupNormSiLU(module[0]), *module[2:])

    return patch_module(m, filter_func, patch_func)


def patch_lora_compatible_conv(m):
    from torch.nn import Conv2d
    # from diffusers.models.lora import LoRACompatibleConv
    from .diffusers import TritonLoRACompatibleConv
    return patch_module(m, lambda stack: isinstance(stack[-1][1], Conv2d),
                        TritonLoRACompatibleConv)


def patch_lora_compatible_linear(m):
    from torch.nn import Linear
    # from diffusers.models.lora import LoRACompatibleLinear
    from .diffusers import TritonLoRACompatibleLinear
    return patch_module(m, lambda stack: isinstance(stack[-1][1], Linear),
                        TritonLoRACompatibleLinear)
