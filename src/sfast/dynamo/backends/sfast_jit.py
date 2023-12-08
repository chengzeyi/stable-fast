import functools
import torch
from torch._dynamo.backends.registry import register_backend
from torch._dynamo.backends.common import aot_autograd, fake_tensor_unsupported
from functorch.compile import make_boxed_compiler
from sfast.jit.trace_helper import trace_with_kwargs


def gen_jit_aot_compiler(compiler, ts_compiler):
    wrapped = functools.partial(compiler, ts_compiler=ts_compiler)
    return make_boxed_compiler(wrapped)


@register_backend
def sfast_jit_script(gm, example_inputs, *, ts_compiler=None, **kwargs):
    ts = torch.jit.script(gm, **kwargs)
    if ts_compiler is not None:
        ts = ts_compiler(ts, example_inputs)
    return ts


@register_backend
def sfast_jit_script_aot_autograd(gm,
                                  example_inputs,
                                  *,
                                  fw_ts_compiler=None,
                                  bw_ts_compiler=None, **kwargs):
    fw_compiler = gen_jit_aot_compiler(
        torch.jit.script, fw_ts_compiler, **kwargs)
    bw_compiler = gen_jit_aot_compiler(
        torch.jit.script, bw_ts_compiler, **kwargs)
    return aot_autograd(fw_compiler=fw_compiler,
                        bw_compiler=bw_compiler)(gm, example_inputs)


@register_backend
@fake_tensor_unsupported
def sfast_jit_trace(gm, example_inputs, *, ts_compiler=None, **kwargs):
    # If check_trace is True, the tracer will run with the example_inputs after tracing.
    # This will cause the GraphFunction executor to generate some cached optimized graphs first.
    # (torch/csrc/jit/api/function_impl.h: get_executor())
    # But we might modify its graph later, so we don't want to cache it.
    # So we set check_trace to False.
    ts, call_helper = trace_with_kwargs(gm,
                                        example_inputs,
                                        **kwargs)
    if ts_compiler is not None:
        ts = ts_compiler(ts, call_helper(ts).convert_inputs(example_inputs))
    return call_helper(ts)


@register_backend
@fake_tensor_unsupported
def sfast_jit_trace_aot_autograd(gm,
                                 example_inputs,
                                 *,
                                 fw_ts_compiler=None,
                                 bw_ts_compiler=None, **kwargs):
    fw_compiler = gen_jit_aot_compiler(
        sfast_jit_trace, fw_ts_compiler, **kwargs)
    bw_compiler = gen_jit_aot_compiler(
        sfast_jit_trace, bw_ts_compiler, **kwargs)
    return aot_autograd(fw_compiler=fw_compiler,
                        bw_compiler=bw_compiler)(gm, example_inputs)
