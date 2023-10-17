import contextlib
import torch


@contextlib.contextmanager
def compute_precision(*, allow_tf32):
    old_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        with torch.backends.cudnn.flags(enabled=None,
                                        benchmark=None,
                                        deterministic=None,
                                        allow_tf32=allow_tf32):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32_matmul


@contextlib.contextmanager
def low_compute_precision():
    try:
        with compute_precision(allow_tf32=True):
            yield
    finally:
        pass


@contextlib.contextmanager
def high_compute_precision():
    try:
        with compute_precision(allow_tf32=False):
            yield
    finally:
        pass
