import logging
import packaging.version
from dataclasses import dataclass
from typing import Union
import functools
import torch
import sfast
from sfast.jit import passes
from sfast.jit.trace_helper import (lazy_trace, to_module)
from sfast.jit import utils as jit_utils
from sfast.cuda.graphs import make_dynamic_graphed_callable
from sfast.utils import gpu_device

logger = logging.getLogger()


class CompilationConfig:

    @dataclass
    class Default:
        memory_format: torch.memory_format = torch.channels_last if gpu_device.device_has_tensor_core(
        ) else torch.contiguous_format
        enable_jit: bool = True
        enable_jit_freeze: bool = True
        enable_cnn_optimization: bool = True
        prefer_lowp_gemm: bool = True
        enable_xformers: bool = False
        enable_cuda_graph: bool = False
        enable_triton: bool = False
        trace_scheduler: bool = False


def compile(m, config):
    with torch.no_grad():
        enable_cuda_graph = config.enable_cuda_graph and m.device.type == 'cuda'

        scheduler = m.scheduler
        scheduler._set_timesteps = scheduler.set_timesteps

        def set_timesteps(self, num_timesteps: int,
                          device: Union[str, torch.device]):
            return self._set_timesteps(num_timesteps,
                                       device=torch.device('cpu'))

        scheduler.set_timesteps = set_timesteps.__get__(scheduler)

        if config.enable_xformers:
            if config.enable_jit:
                from sfast.utils.xformers_attention import xformers_memory_efficient_attention
                from xformers import ops

                ops.memory_efficient_attention = xformers_memory_efficient_attention

            m.enable_xformers_memory_efficient_attention()

        if config.memory_format == torch.channels_last:
            m.unet.to(memory_format=torch.channels_last)
            m.vae.to(memory_format=torch.channels_last)
            if hasattr(m, 'controlnet'):
                m.controlnet.to(memory_format=torch.channels_last)

        if config.enable_jit:
            modify_model = functools.partial(
                _modify_model,
                enable_cnn_optimization=config.enable_cnn_optimization,
                prefer_lowp_gemm=config.prefer_lowp_gemm,
                enable_triton=config.enable_triton,
                memory_format=config.memory_format,
            )

            def ts_compiler(m,
                            call_helper,
                            inputs,
                            kwarg_inputs,
                            freeze=False,
                            enable_cuda_graph=False):
                with torch.jit.optimized_execution(True):
                    if freeze:
                        # raw freeze causes Tensor reference leak
                        # because the constant Tensors in the GraphFunction of
                        # the compilation unit are never freed.
                        m = jit_utils.better_freeze(m)
                    modify_model(m)

                if enable_cuda_graph:
                    m = make_dynamic_graphed_callable(m)
                return m

            lazy_trace_ = functools.partial(
                lazy_trace,
                ts_compiler=functools.partial(
                    ts_compiler,
                    freeze=config.enable_jit_freeze,
                ),
                check_trace=False,
                strict=False)

            # disable jit for text_encoder because of exception caused by
            # tracing BaseModelOutputPooling of StableDiffusionXLPipeline
            m.text_encoder.forward = lazy_trace_(
                to_module(m.text_encoder.forward))
            unet_forward = lazy_trace(to_module(m.unet.forward),
                                      ts_compiler=functools.partial(
                                          ts_compiler,
                                          freeze=config.enable_jit_freeze,
                                          enable_cuda_graph=enable_cuda_graph,
                                      ),
                                      check_trace=False,
                                      strict=False)

            @functools.wraps(m.unet.forward)
            def unet_forward_wrapper(sample, t, *args, **kwargs):
                t = t.to(device=sample.device)
                return unet_forward(sample, t, *args, **kwargs)

            m.unet.forward = unet_forward_wrapper

            if not packaging.version.parse('2.0.0') <= packaging.version.parse(
                    torch.__version__) < packaging.version.parse('2.1.0'):
                '''
                Weird bug in PyTorch 2.0.x

                RuntimeError: shape '[512, 512, 64, 64]' is invalid for input of size 2097152

                When executing AttnProcessor in TorchScript
                '''
                m.vae.decode = lazy_trace_(to_module(m.vae.decode))
                # For img2img
                m.vae.encoder.forward = lazy_trace_(
                    to_module(m.vae.encoder.forward))
                m.vae.quant_conv.forward = lazy_trace_(
                    to_module(m.vae.quant_conv.forward))

            if config.trace_scheduler:
                m.scheduler.scale_model_input = lazy_trace_(
                    to_module(m.scheduler.scale_model_input))
                m.scheduler.step = lazy_trace_(to_module(m.scheduler.step))

            if hasattr(m, 'controlnet'):
                controlnet_forward = lazy_trace(
                    to_module(m.controlnet.forward),
                    ts_compiler=functools.partial(
                        ts_compiler,
                        freeze=False,
                        enable_cuda_graph=enable_cuda_graph,
                    ),
                    check_trace=False,
                    strict=False)

                @functools.wraps(m.controlnet.forward)
                def controlnet_forward_wrapper(sample, t, *args, **kwargs):
                    t = t.to(device=sample.device)
                    return controlnet_forward(sample, t, *args, **kwargs)

                m.controlnet.forward = controlnet_forward_wrapper

        return m


def _modify_model(m,
                  enable_cnn_optimization=True,
                  prefer_lowp_gemm=True,
                  enable_triton=False,
                  memory_format=None):
    if enable_triton:
        from sfast.jit.passes import triton_passes

    torch._C._jit_pass_inline(m.graph)

    passes.jit_pass_remove_dropout(m.graph)

    passes.jit_pass_remove_contiguous(m.graph)
    passes.jit_pass_replace_view_with_reshape(m.graph)
    if enable_triton:
        triton_passes.jit_pass_optimize_reshape(m.graph)

        # triton_passes.jit_pass_optimize_cnn(m.graph)

        triton_passes.jit_pass_fuse_group_norm_silu(m.graph)
        triton_passes.jit_pass_optimize_group_norm(m.graph)

    passes.jit_pass_optimize_linear(m.graph)

    if memory_format is not None:
        sfast._C._jit_pass_convert_op_input_tensors(
            m.graph,
            'aten::_convolution',
            indices=[0],
            memory_format=memory_format)

    if enable_cnn_optimization:
        passes.jit_pass_optimize_cnn(m.graph)

    if prefer_lowp_gemm:
        passes.jit_pass_prefer_lowp_gemm(m.graph)
        passes.jit_pass_fuse_lowp_linear_add(m.graph)
