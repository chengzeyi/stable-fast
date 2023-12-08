import logging
import torch

logger = logging.getLogger()

triton = None
try:
    import triton
except ImportError:
    logger.warning(
        'Triton is not installed, Triton passes will not work properly.')
if triton is not None:
    from sfast.triton import torch_ops


def jit_pass_optimize_cnn(graph):
    jit_pass_optimize_convolution(graph)


def jit_pass_optimize_convolution(graph):
    if hasattr(torch.ops.sfast_triton, '_convolution'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x = aten::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    return (%x)''', '''
graph(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13):
    %x = sfast_triton::_convolution(%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13)
    return (%x)''', graph)


def jit_pass_optimize_contiguous(graph):
    if hasattr(torch.ops.sfast_triton, 'contiguous'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2):
    %x = aten::contiguous(%1, %2)
    return (%x)''', '''
graph(%1, %2):
    %x = sfast_triton::contiguous(%1, %2)
    return (%x)''', graph)


def jit_pass_optimize_reshape(graph):
    if hasattr(torch.ops.sfast_triton, 'reshape'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%1, %2):
    %x = aten::reshape(%1, %2)
    return (%x)''', '''
graph(%1, %2):
    %x = sfast_triton::reshape(%1, %2)
    return (%x)''', graph)


def jit_pass_optimize_group_norm(graph):
    if hasattr(torch.ops.sfast_triton, 'group_norm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %output = aten::group_norm(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled)
    return (%output)''', '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %output = sfast_triton::group_norm(%input, %num_groups, %weight, %bias, %eps)
    return (%output)''', graph)


def jit_pass_fuse_group_norm_silu(graph):
    if hasattr(torch.ops.sfast_triton, 'group_norm_silu'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %x = aten::group_norm(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled)
    %y = aten::silu(%x)
    return (%y)''', '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %y = sfast_triton::group_norm_silu(%input, %num_groups, %weight, %bias, %eps)
    return (%y)''', graph)

        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %x = aten::group_norm(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled)
    %y = aten::silu_(%x)
    return (%y)''', '''
graph(%input, %num_groups, %weight, %bias, %eps, %cudnn_enabled):
    %y = sfast_triton::group_norm_silu(%input, %num_groups, %weight, %bias, %eps)
    return (%y)''', graph)


def jit_pass_optimize_layer_norm(graph):
    if hasattr(torch.ops.sfast_triton, 'layer_norm'):
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(
            '''
graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled):
    %output = aten::layer_norm(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled)
    return (%output)''', '''
graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled):
    %output = sfast_triton::layer_norm(%input, %normalized_shape, %weight, %bias, %eps)
    return (%output)''', graph)
