import pytest

import logging
import torch

logger = logging.getLogger()


class LinearModule(torch.nn.Module):

    def __init__(self, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16, bias=bias)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_dynamic(dtype, bias):
    with torch.no_grad():
        m = LinearModule(bias=bias).eval()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(4, 8)
        out = m_q(x)
        out = out.cuda().to(dtype=dtype)

        m_cuda = m.cuda().to(dtype=dtype)
        m_q_cuda = torch.quantization.quantize_dynamic(
            m_cuda, {torch.nn.Linear},
            dtype=torch.qint8).to(dtype=dtype)
        x_cuda = x.cuda().to(dtype=dtype)
        out_cuda = m_q_cuda(x_cuda)

        logger.info(f"bias={bias}, dtype={dtype}")
        logger.info(f"out={out}")
        logger.info(f"out_cuda={out_cuda}")
