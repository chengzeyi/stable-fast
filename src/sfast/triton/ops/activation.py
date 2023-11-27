import triton
import triton.language as tl


@triton.jit
def identity(x):
    return x


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit
def relu(x):
    return tl.max(x, 0)
