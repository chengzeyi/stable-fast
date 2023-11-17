import pytest

import logging
import torch
from sfast.jit.trace_helper import trace_with_kwargs

logger = logging.getLogger()


class LinearModule(torch.nn.Module):

    def __init__(self, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20, bias=bias)

    def forward(self, x):
        return self.linear(x)


def test_linear_bias():
    m = LinearModule(bias=False).eval().cuda()
    m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                              dtype=torch.qint8)
    x = torch.rand(1, 10).cuda()
    m_q = trace_with_kwargs(m_q, (x,))
    print(m_q.inlined_graph)
