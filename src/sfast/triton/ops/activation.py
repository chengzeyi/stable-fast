import triton
import triton.language as tl


@triton.jit
def identity(x):
    return x


@triton.jit
def silu(x):
    return x * tl.sigmoid(x.to(tl.float32)).to(x.dtype)


@triton.jit
def relu(x):
    return tl.max(x, 0.0)


@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608028654 *
                                    (x + 0.044715 * x * x * x)))
