import functools
import torch
from torch.utils._python_dispatch import _disable_current_modes
from torch._dynamo.backends.registry import register_backend
from torch._subclasses import FakeTensor
from sfast.jit.utils import better_trace


@register_backend
def sfast_jit_script(gm, example_inputs, *, ts_compiler=None):
    ts = torch.jit.script(gm)
    if ts_compiler is not None:
        ts = ts_compiler(ts, example_inputs)
    return ts


def fake_tensor_unsupported(fn):
    """
    Decorator for backends that need real inputs.  We swap out fake
    tensors for zero tensors.
    """

    def defake(x):
        if not isinstance(x, FakeTensor):
            return x
        if x._has_symbolic_sizes_strides:
            size = [s.node.shape_env.size_hint(s.node.expr) for s in x.size()]
            stride = [
                s.node.shape_env.size_hint(s.node.expr) for s in x.stride()
            ]
        else:
            size = x.size()
            stride = x.stride()
        y = torch.empty_strided(
            size,
            stride,
            dtype=x.dtype,
            device=x.device,
            # RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
            requires_grad=False,
            # requires_grad=x.requires_grad,
        )
        y.zero_()
        y.requires_grad = x.requires_grad
        # y.zero_()
        return y

    @functools.wraps(fn)
    def wrapper(model, inputs, **kwargs):
        with _disable_current_modes():
            inputs = list(map(defake, inputs))
            return fn(model, inputs, **kwargs)

    return wrapper


@register_backend
@fake_tensor_unsupported
def sfast_jit_trace(gm, example_inputs, *, ts_compiler=None):
    # If check_trace is True, the tracer will run with the example_inputs after tracing.
    # This will cause the GraphFunction executor to generate some cached optimized graphs first.
    # (torch/csrc/jit/api/function_impl.h: get_executor())
    # But we might modify its graph later, so we don't want to cache it.
    # So we set check_trace to False.
    ts = better_trace(gm, example_inputs, check_trace=False)
    if ts_compiler is not None:
        ts = ts_compiler(ts, example_inputs)
    return ts
