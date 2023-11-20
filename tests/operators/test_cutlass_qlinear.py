import pytest

import logging
import torch

logger = logging.getLogger()


class LinearModule(torch.nn.Module):

    def __init__(self, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32, bias=bias)

    def forward(self, x):
        return self.linear(x)


def test_linear_dynamic():
    with torch.no_grad():
        m = LinearModule().eval().cuda()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(16, 64).cuda()

        torch.testing.assert_allclose(m(x), m_q(x))

        m = LinearModule(bias=False).eval().cuda()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(16, 64).cuda()

        torch.testing.assert_allclose(m(x), m_q(x))

        m = LinearModule().eval().cuda().half()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(16, 64).cuda().half()

        torch.testing.assert_allclose(m(x), m_q(x))

        m = LinearModule(bias=False).eval().cuda().half()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(16, 64).cuda().half()

        torch.testing.assert_allclose(m(x), m_q(x))
