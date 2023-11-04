import logging
import dataclasses
import itertools
import functools
import threading
import copy
import ctypes
import torch
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
    # warmup
    outputs = func(*copy.deepcopy(example_inputs),
                   **copy.deepcopy(example_kwarg_inputs))
    converter = None
    if not isinstance(outputs, (tuple, list)):
        if dataclasses.is_dataclass(outputs):
            converter = DictToDataClassConverter(type(outputs))

    pos_args = convert_to_pos_args(copy.deepcopy(example_inputs),
                                   copy.deepcopy(example_kwarg_inputs))
    traced_module = better_trace(TraceablePosArgOnlyModuleWrapper(func),
                                 pos_args, **kwargs)
    training = getattr(func, 'training', False) if isinstance(
        func, torch.nn.Module) else None
    return traced_module, lambda m: TracedPosArgOnlyModuleWrapper(
        m, training=training, converter=converter)


def lazy_trace(func, *, ts_compiler=None, **kwargs_):
    lock = threading.Lock()
    traced_modules = {}

    @functools.wraps(
        func.forward if isinstance(func, torch.nn.Module) else func)
    def wrapper(*args, **kwargs):
        nonlocal lock, traced_modules
        key = (hash_arg(args), hash_arg(kwargs))
        traced_module = traced_modules.get(key)
        if traced_module is None:
            with lock:
                traced_module = traced_modules.get(key)
                if traced_module is None:
                    logger.info(
                        f'Tracing {getattr(func, "__name__", func.__class__.__name__)}'
                    )
                    traced_m, call_helper = trace_with_kwargs(
                        func, args, kwargs, **kwargs_)
                    if ts_compiler is not None:
                        traced_m = ts_compiler(traced_m, call_helper, args,
                                               kwargs)
                    traced_module = call_helper(traced_m)
                    traced_modules[key] = traced_module
        return traced_module(*args, **kwargs)

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
    if isinstance(arg, (int, float, bool, str, bytes)):
        return arg
    if isinstance(arg, (tuple, list)):
        return tuple(map(hash_arg, arg))
    if isinstance(arg, dict):
        return tuple(
            map(
                hash_arg,
                sorted(((k, hash_arg(v)) for k, v in arg.items()),
                       key=lambda x: x[0])))
    return None


class DictToDataClassConverter:

    def __init__(self, clz):
        self.clz = clz

    def __call__(self, d):
        return self.clz(**d)


class TracedPosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module, *, training=None, converter=None):
        super().__init__()
        self.module = module
        if training is None:
            training = getattr(module, 'training', False) if isinstance(
                module, torch.nn.Module) else False
        self.converter = converter
        self.train(training)

    def forward(self, *args, **kwargs):
        outputs = self.module(*convert_to_pos_args(args, kwargs))
        if self.converter is not None:
            outputs = self.converter(outputs)
        return outputs


class TraceablePosArgOnlyModuleWrapper(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module
        training = getattr(module, 'training', False) if isinstance(
            module, torch.nn.Module) else False
        self.train(training)

    def forward(self, *args):
        pos_arg_num = args[0].item()
        kwargs_arg_num = args[1].item()
        args_to_process = args[2:2 + pos_arg_num * 2]
        kwargs_to_process = args[2 + pos_arg_num * 2:]

        orig_args = []
        for i in range(pos_arg_num):
            orig_args.append(
                convert_from_tensor_arg(args_to_process[i * 2],
                                        args_to_process[i * 2 + 1]))
        orig_kwargs = {}
        for i in range(kwargs_arg_num):
            orig_kwargs[convert_from_tensor_arg(
                kwargs_to_process[i * 4], kwargs_to_process[i * 4 + 1])] = \
                convert_from_tensor_arg(
                    kwargs_to_process[i * 4 + 2], kwargs_to_process[i * 4 + 3])
        return self.module(*orig_args, **orig_kwargs)


def convert_to_pos_args(args, kwargs):
    keys = sorted(kwargs.keys())
    return (torch.tensor(len(args), dtype=torch.int32), ) + (torch.tensor(
        len(keys), dtype=torch.int32), ) + tuple(
            itertools.chain(
                *(convert_to_tensor_arg(arg) for arg in args))) + tuple(
                    itertools.chain(*(convert_to_tensor_arg(key) +
                                      convert_to_tensor_arg(kwargs[key])
                                      for key in keys)))


def convert_to_tensor_arg(arg):
    if arg is None:
        return torch.tensor(0, dtype=torch.int32), torch.Tensor()
    elif isinstance(arg, torch.Tensor):
        return torch.tensor(1, dtype=torch.int32), arg
    elif isinstance(arg, float):
        return torch.tensor(2, dtype=torch.int32), torch.tensor(
            arg, dtype=torch.float64)
    elif isinstance(arg, int):
        return torch.tensor(3,
                            dtype=torch.int32), torch.tensor(arg,
                                                             dtype=torch.int64)
    elif isinstance(arg, bool):
        return torch.tensor(4,
                            dtype=torch.int32), torch.tensor(arg,
                                                             dtype=torch.bool)
    elif isinstance(arg, str):
        return torch.tensor(5, dtype=torch.int32), torch.as_tensor(
            tuple(arg.encode('utf-8')), dtype=torch.uint8)
    elif isinstance(arg, bytes):
        return torch.tensor(6, dtype=torch.int32), torch.as_tensor(
            tuple(arg), dtype=torch.uint8)
    elif isinstance(arg, (list, tuple)):
        return torch.tensor(7, dtype=torch.int32), type(arg)(
            itertools.chain(*(convert_to_tensor_arg(a) for a in arg)))
    elif isinstance(arg, dict):
        keys = sorted(arg.keys())
        return torch.tensor(8, dtype=torch.int32), tuple(
            itertools.chain(*(itertools.chain(convert_to_tensor_arg(key),
                                              convert_to_tensor_arg(arg[key]))
                              for key in keys)))
    else:
        return torch.tensor(
            9, dtype=torch.int32), save_object_reference_in_tensor(arg)


def convert_from_tensor_arg(arg_type, arg):
    arg_type = arg_type.item()
    if arg_type == 0:
        return None
    elif arg_type == 1:
        return arg
    elif arg_type == 2:
        return arg.item()
    elif arg_type == 3:
        return arg.item()
    elif arg_type == 4:
        return arg.item()
    elif arg_type == 5:
        return bytes(arg.tolist()).decode('utf-8')
    elif arg_type == 6:
        return bytes(arg.tolist())
    elif arg_type == 7:
        return type(arg)(convert_from_tensor_arg(t, a)
                         for t, a in zip(arg[::2], arg[1::2]))
    elif arg_type == 8:
        return dict(
            (convert_from_tensor_arg(t1, a1), convert_from_tensor_arg(t2, a2))
            for t1, a1, t2, a2 in zip(arg[::4], arg[1::4], arg[2::4],
                                      arg[3::4]))
    elif arg_type == 9:
        return restore_object_from_tensor(arg)
    else:
        raise ValueError(f"Unknown arg type {arg_type}")


class ObjectStorationHelper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, obj):
        obj_id = id(obj)
        return torch.tensor(obj_id, dtype=torch.int64)

    @staticmethod
    def backward(ctx, grad):
        return None


save_object_reference_in_tensor = ObjectStorationHelper.apply


class ObjectRestorationHelper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        obj_id = t.item()
        return ctypes.cast(obj_id, ctypes.py_object).value

    @staticmethod
    def backward(ctx, grad):
        return None


restore_object_from_tensor = ObjectRestorationHelper.apply
