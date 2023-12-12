import logging
import functools
import threading
import dataclasses
import torch
from sfast.utils.copy import (tree_copy_, tree_copy, shadow_copy,
                              can_be_perfectly_copied)
from sfast.hooks.module_jit_hook import (apply_to_all_modules, apply_to_module)

logger = logging.getLogger()

_per_device_execution_envs = {}
_per_device_execution_envs_lock = threading.Lock()


def make_dynamic_graphed_callable(func):
    lock = threading.Lock()
    cached_callables = {}

    wrapped = func.forward if isinstance(func, torch.nn.Module) else func

    @functools.wraps(wrapped)
    def dynamic_graphed_callable(*args, **kwargs):
        if isinstance(func, torch.nn.Module):
            training = getattr(func, 'training', False)
        elif hasattr(func, '__self__') and isinstance(func.__self__,
                                                      torch.nn.Module):
            training = getattr(func.__self__, 'training', False)
        else:
            training = False
        key = (training, hash_arg(args), hash_arg(kwargs))
        cached_callable = cached_callables.get(key)
        if cached_callable is None:
            with lock:
                cached_callable = cached_callables.get(key)
                if cached_callable is None:
                    logger.info(
                        f'Dynamically graphing {getattr(func, "__name__", func.__class__.__name__)}'
                    )
                    cached_callable = simple_make_graphed_callable(
                        func, args, kwargs)
                    cached_callables[key] = cached_callable
        return cached_callable(*args, **kwargs)

    dynamic_graphed_callable._cached = cached_callables

    return dynamic_graphed_callable


def simple_make_graphed_callable(func,
                                 example_inputs=None,
                                 example_kwarg_inputs=None):
    cuda_device = get_cuda_device_from_tensors(
        (example_inputs, example_kwarg_inputs))
    assert cuda_device is not None
    execution_env = get_per_device_graph_execution_env(cuda_device)
    return make_graphed_callable(func,
                                 example_inputs,
                                 example_kwarg_inputs,
                                 execution_env=execution_env)


def make_graphed_callable(func,
                          example_inputs=None,
                          example_kwarg_inputs=None,
                          *,
                          execution_env):
    is_default_allocator = not hasattr(
        torch.cuda, 'get_allocator_backend'
    ) or torch.cuda.get_allocator_backend() == 'native'

    training = getattr(func, 'training', False) if isinstance(
        func, torch.nn.Module) else False

    if example_inputs is None:
        example_inputs = tuple()
    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream(device=execution_env.device)):
        for _ in range(3):
            func(*tree_copy(example_inputs, detach=True),
                 **tree_copy(example_kwarg_inputs, detach=True))
    torch.cuda.synchronize()

    if is_default_allocator:
        tmp_graph = torch.cuda.CUDAGraph()

        with execution_env.lock:
            with torch.cuda.device(execution_env.device), torch.cuda.stream(
                    execution_env.stream):
                with torch.cuda.graph(tmp_graph,
                                      pool=execution_env.mempool,
                                      stream=execution_env.stream):
                    static_inputs_ = tree_copy(example_inputs)
                    static_kwarg_inputs_ = tree_copy(example_kwarg_inputs)

        static_inputs = shadow_copy(static_inputs_)
        static_kwarg_inputs = shadow_copy(static_kwarg_inputs_)
    else:
        tmp_graph = None
        static_inputs_ = None
        static_kwarg_inputs_ = None

        static_inputs = tree_copy(example_inputs)
        static_kwarg_inputs = tree_copy(example_kwarg_inputs)

    fwd_graph = torch.cuda.CUDAGraph()

    with execution_env.lock:
        with torch.cuda.device(execution_env.device), torch.cuda.stream(
                execution_env.stream):

            with torch.cuda.graph(fwd_graph,
                                  pool=execution_env.mempool,
                                  stream=execution_env.stream):
                static_outputs = func(*static_inputs, **static_kwarg_inputs)

    if is_default_allocator:
        static_outputs = shadow_copy(static_outputs)
    del tmp_graph, static_inputs_, static_kwarg_inputs_

    def make_graphed_function(deps, execution_env, fwd_graph, static_inputs,
                              static_kwarg_inputs, static_outputs, training):

        class _GraphedModule(torch.nn.Module):

            def __init__(self):
                super(_GraphedModule, self).__init__()
                # Hold a reference to the deps so that they don't get GCed
                self.deps = deps
                self.train(training)

            def forward(self, *inputs, **kwarg_inputs):
                with execution_env.lock:
                    outputs = self._forward(*inputs, **kwarg_inputs)
                    outputs = tree_copy(outputs)
                return outputs

            def _forward(self, *inputs, **kwarg_inputs):
                tree_copy_(static_inputs, inputs)
                tree_copy_(static_kwarg_inputs, kwarg_inputs)
                fwd_graph.replay()
                return static_outputs

        _graphed_module = _GraphedModule()

        def functionalized(*user_args, **user_kwarg_args):
            return _graphed_module(*user_args, **user_kwarg_args)

        return functionalized

    def convert_parameter_to_tensor(x):
        if isinstance(x, torch.nn.Parameter):
            return x.data
        return x

    deps = [func]
    if isinstance(func, torch.nn.Module):
        deps.extend(map(convert_parameter_to_tensor, func.parameters()))
    elif hasattr(func, '__self__') and isinstance(func.__self__,
                                                  torch.nn.Module):
        deps.extend(
            map(convert_parameter_to_tensor, func.__self__.parameters()))

    return make_graphed_function(deps,
                                 execution_env,
                                 fwd_graph,
                                 static_inputs,
                                 static_kwarg_inputs,
                                 static_outputs,
                                 training=training)


class GraphExecutionEnv:

    def __init__(self, *, mempool, device=None, stream=None, lock=None):
        self.mempool = mempool
        if isinstance(device, torch.device):
            assert device.type == 'cuda'
            device = device.index
        self.device = torch.cuda.current_device() if device is None else device
        self.stream = torch.cuda.current_stream(
            self.device) if stream is None else stream
        self.lock = threading.Lock() if lock is None else lock

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            with torch.cuda.graph(graph, pool=self.mempool,
                                  stream=self.stream):
                pass
        # Hold a live graph to the mempool so that it has a non-zero use_count
        self.graph = graph


def get_per_device_graph_execution_env(device=None):
    if isinstance(device, torch.device):
        assert device.type == 'cuda'
        device = device.index
    if device is None:
        device = torch.cuda.current_device()
    with _per_device_execution_envs_lock:
        if device not in _per_device_execution_envs:
            with torch.cuda.device(device):
                mempool, stream, lock = torch.cuda.graphs.graph_pool_handle(
                ), torch.cuda.Stream(), threading.Lock()
            _per_device_execution_envs[device] = GraphExecutionEnv(
                mempool=mempool, device=device, stream=stream, lock=lock)
        return _per_device_execution_envs[device]


def hash_arg(arg):
    if isinstance(arg, torch.Tensor):
        arg_device = arg.device
        arg_device_type = arg_device.type
        return (arg_device_type, arg_device.index, arg.dtype, arg.shape,
                arg.item()
                if arg_device_type == 'cpu' and arg.numel() == 1 else None)
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


def get_cuda_device_from_tensors(x):
    if isinstance(x, torch.Tensor):
        device = x.device
        if device.type == 'cuda':
            return device.index
        return None
    elif isinstance(x, (list, tuple)):
        for y in x:
            device = get_cuda_device_from_tensors(y)
            if device is not None:
                return device
        return None
    elif dataclasses.is_dataclass(x):
        for k in dataclasses.fields(x):
            device = get_cuda_device_from_tensors(getattr(x, k))
            if device is not None:
                return device
        return None
    elif isinstance(x, dict):
        for v in x.values():
            device = get_cuda_device_from_tensors(v)
            if device is not None:
                return device
        return None
    else:

        return None


def get_requires_grad_from_tensors(x):
    if isinstance(x, torch.Tensor):
        return x.requires_grad
    elif isinstance(x, (list, tuple)):
        for y in x:
            requires_grad = get_requires_grad_from_tensors(y)
            if requires_grad:
                return True
        return False
    elif dataclasses.is_dataclass(x):
        for k in dataclasses.fields(x):
            requires_grad = get_requires_grad_from_tensors(getattr(x, k))
            if requires_grad:
                return True
        return False
    elif isinstance(x, dict):
        for v in x.values():
            requires_grad = get_requires_grad_from_tensors(v)
            if requires_grad:
                return True
        return False
    else:
        return False


def can_io_obj_be_perfectly_graphed(obj):
    return can_be_perfectly_copied(obj)


class AutoGraphCraphCompiler:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._is_compiling = threading.local()
        self._is_compiling.value = False

    def is_compiling(self):
        return self._is_compiling.value

    def get_inputs_key(self, func, inputs, kwargs):
        if not can_io_obj_be_perfectly_graphed((inputs, kwargs)):
            return None
        return (hash_arg(inputs), hash_arg(kwargs))

    def get_outputs_key(self, func, outputs):
        if not can_io_obj_be_perfectly_graphed(outputs):
            return None
        return hash_arg(outputs)

    def compile(self, func, inputs, kwargs):
        self._is_compiling.value = True
        try:
            graphed = simple_make_graphed_callable(func, inputs, kwargs,
                                                   **self.kwargs)
            wrapped = func.forward if isinstance(func,
                                                 torch.nn.Module) else func

            @functools.wraps(wrapped)
            def functionalized(*args, **kwargs):
                return graphed(*args, **kwargs)

            return functionalized
        finally:
            self._is_compiling.value = False


def apply_auto_graph_compiler_to_all_modules(m, filter_func=None, **kwargs):
    return apply_to_all_modules(m,
                                AutoGraphCraphCompiler(**kwargs),
                                filter_func=filter_func)


def apply_auto_graph_compiler_to_module(m, **kwargs):
    return apply_to_module(m, AutoGraphCraphCompiler(**kwargs))
