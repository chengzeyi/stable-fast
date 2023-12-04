import pytest

import logging
import time
import torch

logger = logging.getLogger()


class LinearModule(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_features', [4, 8, 16])
@pytest.mark.parametrize('out_features', [4, 8, 16])
@pytest.mark.parametrize('N', [4, 16])
def test_linear_dynamic(dtype, bias, in_features, out_features, N):
    with torch.no_grad():
        m = LinearModule(in_features, out_features, bias=bias).eval()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)
        x = torch.randn(N, in_features)
        out = m_q(x)
        out = out.cuda().to(dtype=dtype)

        m_cuda = m.cuda().to(dtype=dtype)
        m_q_cuda = torch.quantization.quantize_dynamic(
            m_cuda, {torch.nn.Linear},
            dtype=torch.qint8).to(dtype=dtype)
        x_cuda = x.cuda().to(dtype=dtype)
        out_cuda = m_q_cuda(x_cuda)

        torch.testing.assert_close(out_cuda, out, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_features', [512, 1024])
@pytest.mark.parametrize('out_features', [512, 1024])
@pytest.mark.parametrize('N', [1, 16, 10000])
def test_benchmark_linear_dynamic(dtype, bias, in_features, out_features, N):
    with torch.no_grad():
        m = LinearModule(in_features, out_features, bias=bias).cuda().to(dtype=dtype).eval()
        m_q = torch.quantization.quantize_dynamic(m, {torch.nn.Linear},
                                                  dtype=torch.qint8)

        x = torch.randn(N, in_features).cuda().to(dtype=dtype)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = m(x)
        torch.cuda.synchronize()
        logger.info(f'cost={time.time() - start}')

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out_q = m_q(x)
        torch.cuda.synchronize()
        logger.info(f'cost={time.time() - start}')
