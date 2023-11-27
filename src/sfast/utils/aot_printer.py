import contextlib
import packaging.version
from functorch.compile import (aot_function, aot_module)
import torch


@contextlib.contextmanager
def no_fake_tensor():
    if packaging.version.parse(
            torch.__version__) >= packaging.version.parse("2.0.0"):
        from torch._functorch import config

        use_fake_tensor = config.use_fake_tensor
        config.use_fake_tensor = False
        try:
            yield
        finally:
            config.use_fake_tensor = use_fake_tensor
    else:
        yield


# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def get_compiler_fn(title=None):

    def compiler_fn(fx_module: torch.fx.GraphModule, _):
        if title is not None:
            print(title)
        print(fx_module.code)
        return fx_module

    return compiler_fn


def aot_printer(fn):
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn,
                          fw_compiler=get_compiler_fn("Forward Code:"),
                          bw_compiler=get_compiler_fn("Backward Code:"))
    else:
        return aot_function(fn,
                            fw_compiler=get_compiler_fn("Forward Code:"),
                            bw_compiler=get_compiler_fn("Backward Code:"))
