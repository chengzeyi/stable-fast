import pytest

import logging
import torch
import torch.nn.functional as F
from sfast.utils.torch_dispatch import (LoggingMode, ReplaceFuncMode)

logger = logging.getLogger()


def test_logging_mode():

    def foo(x, weight, bias):
        return F.group_norm(x, 2, weight, bias, 0.1)

    x = torch.ones(1, 4, 256, 512, device=torch.device('cuda'))
    weight = x.new_zeros(4)
    bias = x.new_zeros(4)

    with LoggingMode():
        foo(x, weight, bias)

    x = x.to(memory_format=torch.channels_last)

    with LoggingMode():
        foo(x, weight, bias)


def test_replace_func_mode():

    def opt_native_group_norm(x, *args, **kwargs):
        return x, None, None

    def foo(x, weight, bias):
        return F.group_norm(x, 2, weight, bias, 0.1)

    x = torch.ones(1, 4, 256, 512, device=torch.device('cuda'))
    weight = x.new_zeros(4)
    bias = x.new_zeros(4)

    with ReplaceFuncMode(
        {torch.ops.aten.native_group_norm.default: opt_native_group_norm},
            debug=True):
        foo(x, weight, bias)
