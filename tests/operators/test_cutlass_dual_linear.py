import pytest

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger()


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int, bias=True):
        super().__init__()
        linear_cls = nn.Linear

        self.proj = linear_cls(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states, enable_opt=False):
        if enable_opt:
            return torch.ops.sfast.cutlass_linear_geglu_unified(
                hidden_states, self.proj.weight, self.proj.bias)
        else:
            hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
            return hidden_states * self.gelu(gate)


@pytest.mark.parametrize('dtype',
                         [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_features', [4, 8, 16])
@pytest.mark.parametrize('out_features', [4, 8, 16])
@pytest.mark.parametrize('N', [4, 16])
def test_geglu(dtype, bias, in_features, out_features, N):
    with torch.no_grad():
        m = GEGLU(in_features, out_features,
                  bias=bias).cuda().to(dtype=dtype).eval()
        x = torch.randn(N, in_features).cuda().to(dtype=dtype)
        out = m(x)
        out_opt = m(x, enable_opt=True)

        torch.testing.assert_close(out_opt, out, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('in_features', [320, 640])
@pytest.mark.parametrize('out_features', [1280, 2560])
@pytest.mark.parametrize('N', [18432 * 25, 4608 * 25])
def test_benchmark_geglu(dtype, bias, in_features, out_features, N):
    with torch.no_grad():
        m = GEGLU(in_features, out_features,
                  bias=bias).cuda().to(dtype=dtype).eval()
        x = torch.randn(N, in_features).cuda().to(dtype=dtype)
        out = m(x)
        out_opt = m(x, enable_opt=True)

        torch.testing.assert_close(out_opt, out, rtol=2e-2, atol=2e-2)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = m(x)
        torch.cuda.synchronize()
        logger.info(f'cost={time.time() - start}')

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out_opt = m(x, enable_opt=True)
        torch.cuda.synchronize()
        logger.info(f'cost={time.time() - start}')
