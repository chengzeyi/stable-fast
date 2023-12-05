import pytest

import logging
import torch
from sfast.jit.trace_helper import (trace_with_kwargs, lazy_trace)

logger = logging.getLogger()


class ConvBiasAddActivation(torch.nn.Module):

    def __init__(self, bias=True, activation_cls=None):
        super(ConvBiasAddActivation, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, bias=bias)
        self.act = activation_cls(
        ) if activation_cls is not None else torch.nn.Identity()

    def forward(self, x, y=None, alpha=1.0, beta_gamma=None, generator=None):
        x = self.conv(x)
        if y is not None:
            x = x.add(y, alpha=alpha)
        x = self.act(x)
        if beta_gamma is not None:
            x = x.add(beta_gamma[0], alpha=beta_gamma[1])
        return x if generator is None else (x, torch.Generator())


def test_trace_with_kwargs():
    with torch.no_grad():
        model = ConvBiasAddActivation(activation_cls=torch.nn.ReLU)
        model.eval()

        x = torch.ones(1, 1, 256, 256)
        y = torch.ones(1, 1, 254, 254)
        args = (x, )
        kwargs = dict(y=y, alpha=0.5, beta_gamma=(1, 0.5))
        traced_model, call_helper = trace_with_kwargs(model, args, kwargs)

        logging.info('Traced graph for eval:\n{}'.format(traced_model.graph))

        traced_model = call_helper(traced_model)
        out = model(*args, **kwargs)
        traced_out = traced_model(*args, **kwargs)

        torch.testing.assert_close(out, traced_out)

    model = ConvBiasAddActivation(activation_cls=torch.nn.ReLU)

    x = torch.ones(1, 1, 256, 256, requires_grad=True)
    y = torch.ones(1, 1, 254, 254, requires_grad=True)
    args = (x, )
    kwargs = dict(y=y, alpha=0.5, beta_gamma=(1, 0.5))
    traced_model, call_helper = trace_with_kwargs(model, args, kwargs)

    logging.info('Traced graph for training:\n{}'.format(traced_model.graph))

    out = model(*args, **kwargs)
    out.sum().backward()
    traced_out = call_helper(traced_model)(*args, **kwargs)
    traced_out.sum().backward()

    torch.testing.assert_close(traced_out, out)
    torch.testing.assert_close(traced_model.module.conv.weight.grad,
                                  model.conv.weight.grad)
    torch.testing.assert_close(traced_model.module.conv.bias.grad,
                                  model.conv.bias.grad)


def test_lazy_trace():
    with torch.no_grad():
        model = ConvBiasAddActivation(activation_cls=torch.nn.ReLU)
        model.eval()

        x = torch.ones(1, 1, 256, 256)
        y = torch.ones(1, 1, 254, 254)
        args = (x, )
        kwargs = dict(y=y, alpha=0.5, beta_gamma=(1, 0.5))

        out = model(*args, **kwargs)

        model = lazy_trace(model)

        traced_out = model(*args, **kwargs)

        torch.testing.assert_close(out, traced_out)


def test_lazy_trace_with_generator():
    with torch.no_grad():
        model = ConvBiasAddActivation(activation_cls=torch.nn.ReLU)
        model.eval()

        x = torch.ones(1, 1, 256, 256)
        y = torch.ones(1, 1, 254, 254)
        args = (x, )
        kwargs = dict(y=y,
                      alpha=0.5,
                      beta_gamma=(1, 0.5),
                      generator=torch.Generator())

        out, g = model(*args, **kwargs)

        model = lazy_trace(model, check_trace=False, strict=False)

        traced_out, traced_g = model(*args, **kwargs)
        # print(tuple(model._cached.values())[0].module.inlined_graph)

        torch.testing.assert_close(out, traced_out)
