import pytest

import gc
import torch
from sfast.cuda.graphs import (simple_make_graphed_callable, get_per_device_graph_execution_env)


def test_simple_make_graphed_callable():
    device = torch.device('cuda')

    def add(x, y):
        return x + y

    x = torch.randn(3, device=device)
    y = torch.randn(3, device=device)
    graphed_add = simple_make_graphed_callable(
        add, example_inputs=(x, y)
    )
    expected_output = add(x, y)
    assert torch.allclose(graphed_add(x, y), expected_output)

    execution_env = get_per_device_graph_execution_env(device.index)
    tmp_graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(execution_env.device), torch.cuda.stream(execution_env.stream):
        with torch.cuda.graph(tmp_graph, pool=execution_env.mempool, stream=execution_env.stream):
            x = torch.randn(3, device=device)
    # Hold a reference to a tensor allocated from the pool
    # so that it doesn't get cleared from graph_pools_freeble in CUDACachingAllocator
    # if its use_count becomes 0.
    # To reproduce the following runtime error because of all graphs using the same pool getting freed:
    # >       super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
    # E       RuntimeError: it->second->use_count > 0 INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":2103, please report a bug to PyTorch.
    del graphed_add, tmp_graph
    gc.collect()

    graphed_add = simple_make_graphed_callable(
        add, example_kwarg_inputs={'x': x, 'y': y}
    )
    expected_output = add(x, y)
    assert torch.allclose(graphed_add(x=x, y=y), expected_output)