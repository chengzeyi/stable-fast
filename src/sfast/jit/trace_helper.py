import logging
import inspect
import functools
import threading
import torch
from sfast.utils import flat_tensors
from sfast.utils.copy import tree_copy
from sfast.hooks.module_jit_hook import (apply_to_all_modules, apply_to_module)
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
    pos_args = flat_tensors.flattern(
        (tree_copy(example_inputs,
                   detach=True), tree_copy(example_kwarg_inputs, detach=True)))
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
        key = (module_to_be_traced.training, hash_arg(args), hash_arg(kwargs))
        traced_module = traced_modules.get(key)
        if traced_module is None:
            with lock:
                traced_module = traced_modules.get(key)
                if traced_module is None:
                    logger.info(f'Tracing {name}')
                    traced_m, call_helper = trace_with_kwargs(
                        module_to_be_traced, args, kwargs, **kwargs_)
                    if ts_compiler is not None:
                        if 'call_helper' in inspect.signature(
                                ts_compiler).parameters:
                            traced_m = ts_compiler(traced_m, call_helper, args,
                                                   kwargs)
                        else:
                            converted_args = call_helper(
                                traced_m).convert_inputs(args, kwargs)
                            traced_m = ts_compiler(traced_m, converted_args)
                    traced_module = call_helper(traced_m)
                    traced_modules[key] = traced_module
        return traced_module(*args, **kwargs)

    if hasattr(func, '__self__'):
        wrapper.__self__ = func.__self__
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

        @property
        def training(self):
            return getattr(self.module, 'training', False)

        # set training status of the module
        @training.setter
        def training(self, mode):
            if hasattr(self, 'module') and hasattr(self.module, 'training'):
                self.module.training = mode

    if self is None and hasattr(func, '__self__') and isinstance(
            func.__self__, torch.nn.Module):
        self = func.__self__
    if self is not None:
        return FuncModule(func, self)
    return FuncModule(func)


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
        outputs = self.module(*self.convert_inputs(args, kwargs))
        unflat_outputs = flat_tensors.unflattern(outputs)
        return unflat_outputs

    def convert_inputs(self, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return flat_tensors.flattern((args, kwargs))


class TraceablePosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        training = getattr(module, 'training', False) if isinstance(
            module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args):
        orig_args, orig_kwargs = flat_tensors.unflattern(args)
        outputs = self.module(*orig_args, **orig_kwargs)
        flat_outputs = flat_tensors.flattern(outputs)
        return flat_outputs


def can_io_obj_be_perfectly_jitted(obj):
    return flat_tensors.can_be_perfectly_flattened(obj)


class AutoJITCompiler:

    def __init__(self, *, ts_compiler=None, **kwargs):
        self.ts_compiler = ts_compiler
        self.kwargs = kwargs
        self._is_compiling = threading.local()
        self._is_compiling.value = False

    def is_compiling(self):
        return self._is_compiling.value

    def get_inputs_key(self, func, inputs, kwargs):
        if not can_io_obj_be_perfectly_jitted((inputs, kwargs)):
            return None
        return (hash_arg(inputs), hash_arg(kwargs))

    def get_outputs_key(self, func, outputs):
        if not can_io_obj_be_perfectly_jitted(outputs):
            return None
        return (hash_arg(outputs), )

    def compile(self, func, inputs, kwargs):
        self._is_compiling.value = True
        try:
            wrapped = func.forward if isinstance(func,
                                                 torch.nn.Module) else func
            module_to_be_traced = to_module(wrapped)
            traced_m, call_helper = trace_with_kwargs(module_to_be_traced,
                                                      inputs, kwargs,
                                                      **self.kwargs)
            if self.ts_compiler is not None:
                if 'call_helper' in inspect.signature(
                        self.ts_compiler).parameters:
                    traced_m = self.ts_compiler(traced_m, call_helper, inputs,
                                                kwargs)
                else:
                    converted_args = call_helper(traced_m).convert_inputs(
                        inputs, kwargs)
                    traced_m = self.ts_compiler(traced_m, converted_args)

            traced_module = call_helper(traced_m)

            @functools.wraps(wrapped)
            def functionalized(*args, **kwargs):
                return traced_module(*args, **kwargs)

            return functionalized
        finally:
            self._is_compiling.value = False


def apply_auto_jit_compiler_to_all_modules(m, filter_func=None, **kwargs):
    return apply_to_all_modules(m,
                                AutoJITCompiler(**kwargs),
                                filter_func=filter_func)


def apply_auto_jit_compiler_to_module(m, **kwargs):
    return apply_to_module(m, AutoJITCompiler(**kwargs))
