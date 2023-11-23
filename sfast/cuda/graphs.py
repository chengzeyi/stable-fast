import logging
import functools
import threading
import copy
import torch
import sfast

logger = logging.getLogger()

_per_device_execution_envs = {}
_per_device_execution_envs_lock = threading.Lock()


def make_dynamic_graphed_callable(callable):
    lock = threading.Lock()
    cached_callables = {}

    @functools.wraps(callable)
    def dynamic_graphed_callable(*args, **kwargs):
        key = (hash_arg(args), hash_arg(kwargs))
        cached_callable = cached_callables.get(key)
        if cached_callable is None:
            with lock:
                cached_callable = cached_callables.get(key)
                if cached_callable is None:
                    logger.info(
                        f'Dynamically graphing {getattr(callable, "__name__", callable.__class__.__name__)}'
                    )
                    cached_callable = simple_make_graphed_callable(
                        callable, args, kwargs)
                    cached_callables[key] = cached_callable
        return cached_callable(*args, **kwargs)

    dynamic_graphed_callable._cached = cached_callables

    return dynamic_graphed_callable


def simple_make_graphed_callable(callable,
                                 example_inputs=None,
                                 example_kwarg_inputs=None):
    cuda_device = get_cuda_device_from_tensors(
        (example_inputs, example_kwarg_inputs))
    assert cuda_device is not None
    execution_env = get_per_device_graph_execution_env(cuda_device)
    return make_graphed_callable(callable,
                                 example_inputs,
                                 example_kwarg_inputs,
                                 execution_env=execution_env)


def make_graphed_callable(callable,
                          example_inputs=None,
                          example_kwarg_inputs=None,
                          *,
                          execution_env):
    is_default_allocator = not hasattr(
        torch.cuda, 'get_allocator_backend'
    ) or torch.cuda.get_allocator_backend() == 'native'

    training = getattr(callable, 'training', False) if isinstance(
        callable, torch.nn.Module) else False

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
            callable(*copy.deepcopy(example_inputs),
                     **copy.deepcopy(example_kwarg_inputs))
    torch.cuda.synchronize()

    if is_default_allocator:
        tmp_graph = torch.cuda.CUDAGraph()

        with execution_env.lock:
            with torch.cuda.device(execution_env.device), torch.cuda.stream(
                    execution_env.stream):
                with torch.cuda.graph(tmp_graph,
                                      pool=execution_env.mempool,
                                      stream=execution_env.stream):
                    static_inputs_ = copy.deepcopy(example_inputs)
                    static_kwarg_inputs_ = copy.deepcopy(example_kwarg_inputs)

        static_inputs = shadow_copy(static_inputs_)
        static_kwarg_inputs = shadow_copy(static_kwarg_inputs_)
    else:
        tmp_graph = None
        static_inputs_ = None
        static_kwarg_inputs_ = None

        static_inputs = copy.deepcopy(example_inputs)
        static_kwarg_inputs = copy.deepcopy(example_kwarg_inputs)

    fwd_graph = torch.cuda.CUDAGraph()

    with execution_env.lock:
        with torch.cuda.device(execution_env.device), torch.cuda.stream(
                execution_env.stream):

            with torch.cuda.graph(fwd_graph,
                                  pool=execution_env.mempool,
                                  stream=execution_env.stream):
                static_outputs = callable(*static_inputs,
                                          **static_kwarg_inputs)

    if is_default_allocator:
        static_outputs = shadow_copy(static_outputs)
    del tmp_graph, static_inputs_, static_kwarg_inputs_

    def make_graphed_function(callable, execution_env, fwd_graph,
                              static_inputs, static_kwarg_inputs,
                              static_outputs, training):

        class _GraphedModule(torch.nn.Module):

            def __init__(self):
                super(_GraphedModule, self).__init__()
                # Hold a reference to the callable so it doesn't get GC'd
                self.callable = callable
                self.train(training)

            def forward(self, *inputs, **kwarg_inputs):
                with execution_env.lock:
                    outputs = self._forward(*inputs, **kwarg_inputs)
                    outputs = copy.deepcopy(outputs)
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

    return make_graphed_function(callable,
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
    return None


def tree_copy_(dest, src):
    if isinstance(dest, torch.Tensor):
        dest.copy_(src)
    elif isinstance(dest, (list, tuple)):
        assert len(dest) == len(src)
        for x, y in zip(dest, src):
            tree_copy_(x, y)
    elif isinstance(dest, dict):
        assert len(dest) == len(src)
        for k in dest:
            tree_copy_(dest[k], src[k])
    else:
        assert dest == src


def tree_copy(src):
    if isinstance(src, torch.Tensor):
        return src.clone()
    elif isinstance(src, (list, tuple)):
        return type(src)(tree_copy(x) for x in src)
    elif isinstance(src, dict):
        return type(src)((k, tree_copy(v)) for k, v in src.items())
    else:
        return src


def shadow_copy(obj):
    if isinstance(obj, torch.Tensor):
        return sfast._C._create_shadow_tensor(
            obj) if obj.device.type == 'cuda' else obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(shadow_copy(x) for x in obj)
    elif isinstance(obj, dict):
        return type(obj)((k, shadow_copy(v)) for k, v in obj.items())
    else:
        return obj


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
    elif isinstance(x, dict):
        for v in x.values():
            device = get_cuda_device_from_tensors(v)
            if device is not None:
                return device
        return None
    else:
        return None
