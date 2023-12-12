import logging
import packaging.version
from dataclasses import dataclass
import functools
import torch
import sfast
from sfast.jit import passes
from sfast.jit.trace_helper import (lazy_trace,
                                    apply_auto_jit_compiler_to_all_modules)
from sfast.jit import utils as jit_utils
from sfast.cuda.graphs import (make_dynamic_graphed_callable,
                               apply_auto_graph_compiler_to_all_modules)
from sfast.utils import gpu_device

logger = logging.getLogger()


class CompilationConfig:

    @dataclass
    class Default:
        '''
        Default compilation config

        memory_format:
            channels_last if tensor core is available, otherwise contiguous_format.
            On GPUs with tensor core, channels_last is faster
        enable_jit:
            Whether to enable JIT, most optimizations are done with JIT
        enable_jit_freeze:
            Whether to freeze the model after JIT tracing.
            Freezing the model will enable us to optimize the model further.
        preserve_parameters:
            Whether to preserve parameters when freezing the model.
            If True, parameters will be preserved, but the model will be a bit slower.
            If False, parameters will be marked as constants, and the model will be faster.
            However, if parameters are not preserved, LoRA cannot be switched dynamically.
        enable_cnn_optimization:
            Whether to enable CNN optimization by fusion.
        enable_fused_linear_geglu:
            Whether to enable fused Linear-GEGLU kernel.
        prefer_lowp_gemm:
            Whether to prefer low-precision GEMM and a series of fusion optimizations.
            This will make the model faster, but may cause numerical issues.
        enable_xformers:
            Whether to enable xformers and hijack it to make it compatible with JIT tracing.
        enable_cuda_graph:
            Whether to enable CUDA graph. CUDA Graph will significantly speed up the model,
            by reducing the overhead of CUDA kernel launch, memory allocation, etc.
            However, it will also increase the memory usage.
            Our implementation of CUDA graph supports dynamic shape by caching graphs of
            different shapes.
        enable_triton:
            Whether to enable Triton generated CUDA kernels.
            Triton generated CUDA kernels are faster than PyTorch's CUDA kernels.
            However, Triton has a lot of bugs, and can increase the CPU overhead,
            though the overhead can be reduced by enabling CUDA graph.
        trace_scheduler:
            Whether to trace the scheduler.
        '''
        memory_format: torch.memory_format = (
            torch.channels_last if gpu_device.device_has_tensor_core() else
            torch.contiguous_format)
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        preserve_parameters: bool = True
        enable_cnn_optimization: bool = gpu_device.device_has_tensor_core()
        enable_fused_linear_geglu: bool = gpu_device.device_has_capability(
            8, 0)
        prefer_lowp_gemm: bool = True
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False
        trace_scheduler: bool = False


def compile(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    m.unet = compile_unet(m.unet, config)
    if hasattr(m, 'controlnet'):
        m.controlnet = compile_unet(m.controlnet, config)
    m.vae = compile_vae(m.vae, config)

    if config.enable_jit:
        lazy_trace_ = _build_lazy_trace(config)

        # SVD doesn't have a text encoder
        if getattr(m, 'text_encoder', None) is not None:
            m.text_encoder.forward = lazy_trace_(m.text_encoder.forward)
        if getattr(m, 'text_encoder_2', None) is not None:
            m.text_encoder_2.forward = lazy_trace_(m.text_encoder_2.forward)
        if config.trace_scheduler:
            m.scheduler.scale_model_input = lazy_trace_(
                m.scheduler.scale_model_input)
            m.scheduler.step = lazy_trace_(m.scheduler.step)

    if enable_cuda_graph:
        if getattr(m, 'text_encoder', None) is not None:
            m.text_encoder.forward = make_dynamic_graphed_callable(
                m.text_encoder.forward)
        if getattr(m, 'text_encoder_2', None) is not None:
            m.text_encoder_2.forward = make_dynamic_graphed_callable(
                m.text_encoder_2.forward)

    if hasattr(m, 'image_processor'):
        from sfast.libs.diffusers.image_processor import patch_image_prcessor
        patch_image_prcessor(m.image_processor)

    return m


def compile_unet(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    if config.enable_xformers:
        _enable_xformers(m)

    if config.memory_format is not None:
        m.to(memory_format=config.memory_format)

    if config.enable_jit:
        lazy_trace_ = _build_lazy_trace(
            config,
            enable_triton_reshape=enable_cuda_graph,
            enable_triton_layer_norm=enable_cuda_graph,
        )
        m.forward = lazy_trace_(m.forward)

    if enable_cuda_graph:
        m.forward = make_dynamic_graphed_callable(m.forward)

    return m


def compile_vae(m, config):
    # attribute `device` is not generally available
    device = m.device if hasattr(m, 'device') else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    enable_cuda_graph = config.enable_cuda_graph and device.type == 'cuda'

    if config.enable_xformers:
        _enable_xformers(m)

    if config.memory_format is not None:
        m.to(memory_format=config.memory_format)

    if config.enable_jit:
        if (not packaging.version.parse('2.0.0') <= packaging.version.parse(
                torch.__version__) < packaging.version.parse('2.1.0')):
            """
            Weird bug in PyTorch 2.0.x

            RuntimeError: shape '[512, 512, 64, 64]' is invalid for input of size 2097152

            When executing AttnProcessor in TorchScript
            """
            ts_compiler = _build_ts_compiler(
                config,
                enable_triton_reshape=enable_cuda_graph,
                enable_triton_layer_norm=enable_cuda_graph,
            )
            m = apply_auto_jit_compiler_to_all_modules(m,
                                                       ts_compiler=ts_compiler)

    # if enable_cuda_graph:
    #     m = apply_auto_graph_compiler_to_all_modules(m)

    return m


def _modify_model(
    m,
    enable_cnn_optimization=True,
    enable_fused_linear_geglu=True,
    prefer_lowp_gemm=True,
    enable_triton=False,
    enable_triton_reshape=False,
    enable_triton_layer_norm=False,
    memory_format=None,
):
    if enable_triton:
        from sfast.jit.passes import triton_passes

    training = getattr(m, 'training', False)

    torch._C._jit_pass_inline(m.graph)
    '''
    RuntimeError: 0 INTERNAL ASSERT FAILED at "../torch/csrc/jit/ir/alias_analysis.cpp":616, please report a bug to PyTorch. We don't have an op for aten::to but it isn't a special case.  Argument types: int, Device, int, bool, bool, NoneType,
    '''
    # sfast._C._jit_pass_erase_scalar_tensors(m.graph)
    sfast._C._jit_pass_eliminate_simple_arith(m.graph)

    # passes.jit_pass_prefer_tanh_approx_gelu(m.graph)

    if not training:
        passes.jit_pass_remove_dropout(m.graph)

    passes.jit_pass_remove_contiguous(m.graph)
    passes.jit_pass_replace_view_with_reshape(m.graph)
    if enable_triton:
        if enable_triton_reshape:
            triton_passes.jit_pass_optimize_reshape(m.graph)

        # triton_passes.jit_pass_optimize_cnn(m.graph)

        triton_passes.jit_pass_fuse_group_norm_silu(m.graph)
        triton_passes.jit_pass_optimize_group_norm(m.graph)

        if enable_triton_layer_norm:
            triton_passes.jit_pass_optimize_layer_norm(m.graph)

    if enable_fused_linear_geglu and not training:
        passes.jit_pass_fuse_linear_geglu(m.graph)

    if not training:
        passes.jit_pass_optimize_linear(m.graph)

    if memory_format is not None:
        sfast._C._jit_pass_convert_op_input_tensors(
            m.graph,
            'aten::_convolution',
            indices=[0],
            memory_format=memory_format)

    if enable_cnn_optimization:
        passes.jit_pass_optimize_cnn(m.graph)

    if prefer_lowp_gemm and not training:
        passes.jit_pass_prefer_lowp_gemm(m.graph)
        passes.jit_pass_fuse_lowp_linear_add(m.graph)


def _ts_compiler(
    m,
    inputs,
    modify_model_fn=None,
    freeze=False,
    preserve_parameters=False,
):
    with torch.jit.optimized_execution(True):
        if freeze and not getattr(m, 'training', False):
            # raw freeze causes Tensor reference leak
            # because the constant Tensors in the GraphFunction of
            # the compilation unit are never freed.
            m = jit_utils.better_freeze(
                m,
                preserve_parameters=preserve_parameters,
            )

        if modify_model_fn is not None:
            modify_model_fn(m)

    return m


def _build_lazy_trace(config,
                      enable_triton_reshape=False,
                      enable_triton_layer_norm=False):

    lazy_trace_ = functools.partial(
        lazy_trace,
        ts_compiler=_build_ts_compiler(
            config,
            enable_triton_reshape=enable_triton_reshape,
            enable_triton_layer_norm=enable_triton_layer_norm),
        check_trace=False,
        strict=False,
    )

    return lazy_trace_


def _build_ts_compiler(config,
                       enable_triton_reshape=False,
                       enable_triton_layer_norm=False):
    modify_model = functools.partial(
        _modify_model,
        enable_cnn_optimization=config.enable_cnn_optimization,
        enable_fused_linear_geglu=config.enable_fused_linear_geglu,
        prefer_lowp_gemm=config.prefer_lowp_gemm,
        enable_triton=config.enable_triton,
        enable_triton_reshape=enable_triton_reshape,
        enable_triton_layer_norm=enable_triton_layer_norm,
        memory_format=config.memory_format,
    )

    ts_compiler = functools.partial(
        _ts_compiler,
        freeze=config.enable_jit_freeze,
        preserve_parameters=config.preserve_parameters,
        modify_model_fn=modify_model,
    )

    return ts_compiler


def _enable_xformers(m):
    from xformers import ops
    from sfast.libs.xformers.xformers_attention import xformers_memory_efficient_attention

    ops.memory_efficient_attention = xformers_memory_efficient_attention

    if hasattr(m, 'enable_xformers_memory_efficient_attention'):
        m.enable_xformers_memory_efficient_attention()

        if isinstance(m, torch.nn.Module):
            from sfast.libs.diffusers.xformers_attention import patch_all_attention_modules

            patch_all_attention_modules(m)
    else:
        logger.warning(
            'enable_xformers_memory_efficient_attention() is not available.'
            ' If you have enabled xformers by other means, ignore this warning.'
        )
