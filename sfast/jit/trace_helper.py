import logging
import functools
import threading
import copy
import torch
from sfast.utils.flat_tensors import (convert_to_flat_tensors,
                                      convert_from_flat_tensors)
from .utils import better_trace

logger = logging.getLogger()


def trace_with_kwargs(func,
                      example_inputs=None,
                      example_kwarg_inputs=None,
                      **kwargs):
    if example_inputs is None:
        example_inputs = tuple()
    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}
    pos_args = convert_to_flat_tensors(
        (copy.deepcopy(example_inputs), copy.deepcopy(example_kwarg_inputs)))
    traced_module = better_trace(TraceablePosArgOnlyModuleWrapper(func),
                                 pos_args, **kwargs)
    training = getattr(func, 'training', False) if isinstance(
        func, torch.nn.Module) else None
    return traced_module, lambda m: TracedPosArgOnlyModuleWrapper(
        m, training=training)


def lazy_trace(func, *, ts_compiler=None, **kwargs_):
    lock = threading.Lock()
    traced_modules = {}

    name = getattr(func, '__name__', func.__class__.__name__)
    wrapped = func.forward if isinstance(func, torch.nn.Module) else func
    module_to_be_traced = to_module(wrapped)

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        nonlocal lock, traced_modules
        key = (hash_arg(args), hash_arg(kwargs))
        traced_module = traced_modules.get(key)
        if traced_module is None:
            with lock:
                traced_module = traced_modules.get(key)
                if traced_module is None:
                    logger.info(f'Tracing {name}')
                    traced_m, call_helper = trace_with_kwargs(
                        module_to_be_traced, args, kwargs, **kwargs_)
                    if ts_compiler is not None:
                        traced_m = ts_compiler(traced_m, call_helper, args,
                                               kwargs)
                    traced_module = call_helper(traced_m)
                    traced_modules[key] = traced_module
        return traced_module(*args, **kwargs)

    wrapper._cached = traced_modules

    return wrapper


def to_module(func, self=None):
    if isinstance(func, torch.nn.Module):
        return func

    class FuncModule(torch.nn.Module):

        def __init__(self, func, module=None):
            super().__init__()
            self.func = func
            self.module = module
            self.__name__ = func.__name__

        @functools.wraps(func)
        def forward(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    if self is None and hasattr(func, '__self__') and isinstance(
            func.__self__, torch.nn.Module):
        self = func.__self__
    if self is not None:
        return FuncModule(func, self).train(self.training)
    return FuncModule(func).eval()


def hash_arg(arg):
    # micro optimization: bool obj is an instance of int
    if isinstance(arg, (str, int, float, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            sorted(((hash_arg(k), hash_arg(v)) for k, v in arg.items()),
                   key=lambda x: x[0]))
    return type(arg)


class TracedPosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module, *, training=None):
        super().__init__()
        self.module = module
        if training is None:
            training = getattr(module, 'training', False) if isinstance(
                module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args, **kwargs):
        outputs = self.module(*convert_to_flat_tensors((args, kwargs)))
        unflat_outputs = convert_from_flat_tensors(outputs)
        return unflat_outputs


class TraceablePosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        training = getattr(module, 'training', False) if isinstance(
            module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args):
        orig_args, orig_kwargs = convert_from_flat_tensors(args)
        outputs = self.module(*orig_args, **orig_kwargs)
        flat_outputs = convert_to_flat_tensors(outputs)
        return flat_outputs
