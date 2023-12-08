import torch
import sfast
from sfast.utils.custom_python_operator import register_custom_python_operator
from .ops.copy import copy
from .ops.group_norm import (group_norm_forward, group_norm_silu_forward)
from .ops.layer_norm import LayerNorm as TritonLayerNorm
from .ops.conv import conv_forward

aten = torch.ops.aten


def construct_triton_contiguous_torch_op():

    class TritonContiguous(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, memory_format=torch.contiguous_format):
            if x.device.type != 'cuda' or x.ndim > 4 or x.is_contiguous(
                    memory_format=memory_format):
                return aten.contiguous(x, memory_format=memory_format)
            else:
                dst = torch.empty_like(x, memory_format=memory_format)
                return copy(dst, x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def contiguous(x, memory_format=torch.contiguous_format):
        return TritonContiguous.apply(x, memory_format)

    return contiguous


contiguous = construct_triton_contiguous_torch_op()
register_custom_python_operator(
    'sfast_triton::contiguous(Tensor a, MemoryFormat memory_format) -> Tensor',
    contiguous)


def constuct_triton_clone_torch_op():

    class TritonClone(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, memory_format=torch.preserve_format):
            if x.device.type != 'cuda' or x.ndim > 4 or x.is_contiguous(
                    memory_format=memory_format):
                return aten.clone(x, memory_format=memory_format)
            else:
                dst = torch.empty_like(x, memory_format=memory_format)
                return copy(dst, x)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output, None

    def clone(x, memory_format=torch.preserve_format):
        return TritonClone.apply(x, memory_format)

    return clone


clone = constuct_triton_clone_torch_op()
register_custom_python_operator(
    'sfast_triton::clone(Tensor a, MemoryFormat memory_format) -> Tensor',
    clone)


def construct_triton_reshape_torch_op():

    class TritonReshape(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, shape):
            ctx.shape = x.shape
            if x.device.type != 'cuda' or x.ndim > 4 or sfast._C._compute_stride(
                    x.shape, x.stride(), shape) is not None:
                return aten.reshape(x, shape)
            else:
                dst = torch.empty_like(x,
                                       memory_format=torch.contiguous_format)
                copy(dst, x)
                return aten.reshape(dst, shape)

        @staticmethod
        def backward(ctx, grad_output):
            if grad_output.device.type != 'cuda' or grad_output.ndim > 4 or sfast._C._compute_stride(
                    grad_output.shape, grad_output.stride(),
                    ctx.shape) is not None:
                return grad_output.reshape(ctx.shape), None
            else:
                dst = torch.empty_like(grad_output,
                                       memory_format=torch.contiguous_format)
                copy(dst, grad_output)
                return dst.reshape(ctx.shape), None

    def reshape(x, shape):
        return TritonReshape.apply(x, shape)

    return reshape


reshape = construct_triton_reshape_torch_op()
register_custom_python_operator(
    'sfast_triton::reshape(Tensor a, int[] shape) -> Tensor', reshape)


def construct_triton_group_norm_torch_op():

    class TritonGroupNorm(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, num_groups, weight=None, bias=None, eps=1e-05):
            device_type = input.device.type
            if device_type != 'cuda' or input.ndim > 4:
                input = input.contiguous()
                if weight is not None:
                    weight = weight.contiguous()
                if bias is not None:
                    bias = bias.contiguous()
                N, C = input.shape[:2]
                HxW = input.numel() // (N * C)
                output, mean, rstd = aten.native_group_norm(
                    input, weight, bias, N, C, HxW, num_groups, eps)
            else:
                needs_backward = any(x is not None and x.requires_grad
                                     for x in [input, weight, bias])
                output, mean, rstd = group_norm_forward(
                    input,
                    num_groups,
                    weight,
                    bias,
                    eps,
                    output_mean=needs_backward,
                    output_rstd=needs_backward)
            ctx.save_for_backward(input, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, mean, rstd = ctx.saved_tensors
            grad_input_mask = (ctx.needs_input_grad[0],
                               ctx.needs_input_grad[2],
                               ctx.needs_input_grad[3])
            N, C = input.shape[:2]
            HxW = input.numel() // (N * C)
            grad_output = grad_output.contiguous()
            input = input.contiguous()
            mean = mean.contiguous()
            rstd = rstd.contiguous()
            weight = weight.contiguous() if weight is not None else None
            grad_inputs = aten.native_group_norm_backward(
                grad_output, input, mean, rstd, weight, N, C, HxW,
                ctx.num_groups, grad_input_mask)
            grad_input, grad_weight, grad_bias = grad_inputs
            return grad_input, None, grad_weight, grad_bias, None

    def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
        return TritonGroupNorm.apply(input, num_groups, weight, bias, eps)

    return group_norm


group_norm = construct_triton_group_norm_torch_op()
register_custom_python_operator(
    'sfast_triton::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps) -> Tensor',
    group_norm)


def construct_triton_group_norm_silu_torch_op():

    class TritonGroupNormSiLU(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, num_groups, weight=None, bias=None, eps=1e-05):
            device_type = input.device.type
            if device_type != 'cuda' or input.ndim > 4:
                input = input.contiguous()
                if weight is not None:
                    weight = weight.contiguous()
                if bias is not None:
                    bias = bias.contiguous()
                N, C = input.shape[:2]
                HxW = input.numel() // (N * C)
                output, mean, rstd = aten.native_group_norm(
                    input, weight, bias, N, C, HxW, num_groups, eps)
                output = aten.silu(output)
            else:
                needs_backward = any(x is not None and x.requires_grad
                                     for x in [input, weight, bias])
                output, mean, rstd = group_norm_silu_forward(
                    input,
                    num_groups,
                    weight,
                    bias,
                    eps,
                    output_mean=needs_backward,
                    output_rstd=needs_backward)
            ctx.save_for_backward(input, weight, bias, mean, rstd)
            ctx.num_groups = num_groups
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias, mean, rstd = ctx.saved_tensors
            grad_input_mask = (ctx.needs_input_grad[0],
                               ctx.needs_input_grad[2],
                               ctx.needs_input_grad[3])
            N, C = input.shape[:2]
            HxW = input.numel() // (N * C)
            grad_output = grad_output.contiguous()
            input = input.contiguous()
            mean = mean.contiguous()
            rstd = rstd.contiguous()
            weight = weight.contiguous() if weight is not None else None
            repeats = input.shape[1] // ctx.num_groups
            normed = input.sub(
                mean.repeat_interleave(repeats, 1)[..., None, None]).mul_(
                    rstd.repeat_interleave(repeats, 1)[..., None, None])
            grad_normed = aten.silu_backward(grad_output, normed)
            grad_inputs = aten.native_group_norm_backward(
                grad_normed, input, mean, rstd, weight, N, C, HxW,
                ctx.num_groups, grad_input_mask)
            grad_input, grad_weight, grad_bias = grad_inputs
            return grad_input, None, grad_weight, grad_bias, None

    def group_norm_silu(input, num_groups, weight=None, bias=None, eps=1e-05):
        return TritonGroupNormSiLU.apply(input, num_groups, weight, bias, eps)

    return group_norm_silu


group_norm_silu = construct_triton_group_norm_silu_torch_op()
register_custom_python_operator(
    'sfast_triton::group_norm_silu(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps) -> Tensor',
    group_norm_silu)


def construct_triton_layer_norm_torch_op():

    def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
        if input.device.type != 'cuda' or not input.is_contiguous():
            return aten.layer_norm(input, normalized_shape, weight, bias, eps)
        return TritonLayerNorm.apply(input, normalized_shape, weight, bias,
                                     eps)

    return layer_norm


layer_norm = construct_triton_layer_norm_torch_op()
register_custom_python_operator(
    'sfast_triton::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> Tensor',
    layer_norm)


def construct__convolution_torch_op():

    class Triton_Convolution(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, weight, bias, stride, padding, dilation,
                    transposed, output_padding, groups, benchmark,
                    deterministic, cudnn_enabled, allow_tf32):
            if groups != 1 or transposed or deterministic or not allow_tf32:
                output = aten._convolution(input, weight, bias, stride,
                                           padding, dilation, transposed,
                                           output_padding, groups, benchmark,
                                           deterministic, cudnn_enabled,
                                           allow_tf32)
            else:
                output = conv_forward(input, weight, bias, stride, padding,
                                      dilation, transposed, output_padding,
                                      groups)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            return None, None, None, None, None, None, None, None, None, None, None, None, None

    def _convolution(input, weight, bias, stride, padding, dilation,
                     transposed, output_padding, groups, benchmark,
                     deterministic, cudnn_enabled, allow_tf32):
        return Triton_Convolution.apply(input, weight, bias, stride, padding,
                                        dilation, transposed, output_padding, groups,
                                        benchmark, deterministic, cudnn_enabled,
                                        allow_tf32)

    return _convolution


_convolution = construct__convolution_torch_op()
register_custom_python_operator(
    'sfast_triton::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor',
    _convolution)
