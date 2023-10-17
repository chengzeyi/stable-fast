import threading
import torch

_per_device_execution_envs = {}
_per_device_execution_envs_lock = threading.Lock()


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
    training = getattr(callable, 'training', False) if isinstance(
        callable, torch.nn.Module) else False

    if example_inputs is None:
        example_inputs = tuple()
    if example_kwarg_inputs is None:
        example_kwarg_inputs = dict()

    fwd_graph = torch.cuda.CUDAGraph()

    # Warmup
    # Hopefully prevents cudnn benchmarking and other lazy-initialization cuda work
    # from ending up in any captures.
    torch.cuda.synchronize()
    with torch.cuda.stream(torch.cuda.Stream(device=execution_env.device)):
        for _ in range(3):
            callable(*tree_copy(example_inputs),
                     **tree_copy(example_kwarg_inputs))
    torch.cuda.synchronize()

    static_inputs = tree_copy(example_inputs)
    static_kwarg_inputs = tree_copy(example_kwarg_inputs)

    with execution_env.lock:
        with torch.cuda.device(execution_env.device), torch.cuda.stream(
                execution_env.stream):
            with torch.cuda.graph(fwd_graph,
                                  pool=execution_env.mempool,
                                  stream=execution_env.stream):
                static_outputs = callable(*static_inputs,
                                          **static_kwarg_inputs)

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

    return make_graphed_function(callable,
                                 execution_env,
                                 fwd_graph,
                                 static_inputs,
                                 static_kwarg_inputs,
                                 static_outputs,
                                 training=training)


def tree_copy_(dest, src):
    if isinstance(dest, torch.Tensor):
        dest.copy_(src)
    elif isinstance(dest, (list, tuple)):
        for x, y in zip(dest, src):
            tree_copy_(x, y)
    elif isinstance(dest, dict):
        for k in dest:
            tree_copy_(dest[k], src[k])
    else:
        assert dest == src


def tree_copy(src):
    if isinstance(src, torch.Tensor):
        return src.detach().clone()
    elif isinstance(src, (list, tuple)):
        return type(src)(tree_copy(x) for x in src)
    elif isinstance(src, dict):
        return type(src)((k, tree_copy(v)) for k, v in src.items())
    else:
        return src


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
