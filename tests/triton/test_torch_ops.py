import pytest

import logging
import torch

logger = logging.getLogger()

sfast_triton = torch.ops.sfast_triton


def test_triton_contiguous_torch_op():

    def call_triton_contiguous(x, memory_format=torch.contiguous_format):
        return sfast_triton.contiguous(x, memory_format=memory_format)

    a = torch.ones(1, 4, 256, 512).cuda().permute(0, 1, 3, 2)

    out = call_triton_contiguous(a)
    assert out.is_contiguous()
    torch.testing.assert_close(out, a)

    out = call_triton_contiguous(a, memory_format=torch.channels_last)
    assert out.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(out, a)

    traced_triton_contiguous = torch.jit.trace(call_triton_contiguous, (a,))
    out = traced_triton_contiguous(a)
    assert out.is_contiguous()
    torch.testing.assert_close(out, a)
    logger.info(f'Graph: {traced_triton_contiguous.graph}')
