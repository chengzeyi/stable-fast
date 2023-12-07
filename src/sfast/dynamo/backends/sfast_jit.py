import functools
import torch
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd, fake_tensor_unsupported
from sfast.jit.utils import better_trace


@register_backend
def sfast_jit_script(gm, example_inputs, *, ts_compiler=None):
    ts = torch.jit.script(gm)
    if ts_compiler is not None:
        ts = ts_compiler(ts, example_inputs)
    return ts


@register_backend
def sfast_jit_script_aot_autograd(gm,
                                  example_inputs,
                                  *,
                                  fw_ts_compiler=None,
                                  bw_ts_compiler=None):
    fw_compiler = functools.partial(sfast_jit_script,
                                    ts_compiler=fw_ts_compiler)
    bw_compiler = functools.partial(sfast_jit_script,
                                    ts_compiler=bw_ts_compiler)
    return aot_autograd(fw_compiler=fw_compiler,
                        bw_compiler=bw_compiler)(gm, example_inputs)


@register_backend
@fake_tensor_unsupported
def sfast_jit_trace(gm, example_inputs, *, ts_compiler=None):
    # If check_trace is True, the tracer will run with the example_inputs after tracing.
    # This will cause the GraphFunction executor to generate some cached optimized graphs first.
    # (torch/csrc/jit/api/function_impl.h: get_executor())
    # But we might modify its graph later, so we don't want to cache it.
    # So we set check_trace to False.
    ts = better_trace(gm, example_inputs, check_trace=False, strict=False)
    if ts_compiler is not None:
        ts = ts_compiler(ts, example_inputs)
    return ts


@register_backend
@fake_tensor_unsupported
def sfast_jit_trace_aot_autograd(gm,
                                 example_inputs,
                                 *,
                                 fw_ts_compiler=None,
                                 bw_ts_compiler=None):
    fw_compiler = functools.partial(sfast_jit_trace,
                                    ts_compiler=fw_ts_compiler)
    bw_compiler = functools.partial(sfast_jit_trace,
                                    ts_compiler=bw_ts_compiler)
    return aot_autograd(fw_compiler=fw_compiler,
                        bw_compiler=bw_compiler)(gm, example_inputs)
