import pytest

import logging
import copy
import torch
from sfast.utils.aot_printer import aot_printer

logger = logging.getLogger()


class ConvBiasAddActivation(torch.nn.Module):

    def __init__(self, bias=True, activation_cls=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 3, bias=bias)
        self.act = activation_cls(
        ) if activation_cls is not None else torch.nn.Identity()

    def forward(self, x, y=None, alpha=1.0):
        x = self.conv(x)
        if y is not None:
            x = x.add(y, alpha=alpha)
        x = self.act(x)
        return x


class FusedConvBiasAddActivation(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.conv = m.conv
        self.act = m.act

        self.train(m.training)

    def forward(self, x, y=None, alpha=1.0):
        raise NotImplementedError()


def test_conv_bias_add():

    class FusedConvBiasAdd(FusedConvBiasAddActivation):

        def forward(self, x, y=None, alpha=1.0):
            conv = self.conv
            return torch.ops.sfast.cudnn_convolution_bias_add(
                x, conv.weight, conv.bias, y, alpha, conv.stride, conv.padding,
                conv.dilation, conv.transposed, conv.output_padding,
                conv.groups)

    orig_model = ConvBiasAddActivation()
    orig_model.cuda()

    model = copy.deepcopy(orig_model)
    x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    y = torch.ones(1, 1, 254, 254, requires_grad=True).cuda()
    out = aot_printer(model)(x, y, 0.5)
    out.sum().backward()

    fused_model = FusedConvBiasAdd(copy.deepcopy(orig_model))
    fused_x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    fused_y = torch.ones(1, 1, 254, 254, requires_grad=True).cuda()
    fused_out = aot_printer(fused_model)(fused_x, fused_y, 0.5)
    fused_out.sum().backward()

    torch.testing.assert_close(fused_out, out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_model.conv.weight.grad,
                               model.conv.weight.grad)
    torch.testing.assert_close(fused_model.conv.bias.grad,
                               model.conv.bias.grad)


def test_conv_bias():

    class FusedConvBiasAdd(FusedConvBiasAddActivation):

        def forward(self, x, y=None, alpha=1.0):
            conv = self.conv
            return torch.ops.sfast.cudnn_convolution_bias(
                x, conv.weight,  conv.bias, conv.stride, conv.padding,
                conv.dilation, conv.transposed, conv.output_padding,
                conv.groups)

    orig_model = ConvBiasAddActivation()
    orig_model.cuda()

    model = copy.deepcopy(orig_model)
    x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    out = aot_printer(model)(x)
    out.sum().backward()

    fused_model = FusedConvBiasAdd(copy.deepcopy(orig_model))
    fused_x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    fused_out = aot_printer(fused_model)(fused_x)
    fused_out.sum().backward()

    torch.testing.assert_close(fused_out, out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_model.conv.weight.grad,
                               model.conv.weight.grad)
    torch.testing.assert_close(fused_model.conv.bias.grad,
                               model.conv.bias.grad)


def test_conv_bias_sigmoid():

    class FusedConvBiasAdd(FusedConvBiasAddActivation):

        def forward(self, x, y=None, alpha=1.0):
            conv = self.conv
            return torch.ops.sfast.cudnn_convolution_bias_sigmoid(
                x, conv.weight,  conv.bias, conv.stride, conv.padding,
                conv.dilation, conv.transposed, conv.output_padding,
                conv.groups)

    orig_model = ConvBiasAddActivation(activation_cls=torch.nn.Sigmoid)
    orig_model.cuda()

    model = copy.deepcopy(orig_model)
    x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    out = aot_printer(model)(x)
    out.sum().backward()

    fused_model = FusedConvBiasAdd(copy.deepcopy(orig_model))
    fused_x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    fused_out = aot_printer(fused_model)(fused_x)
    fused_out.sum().backward()

    torch.testing.assert_close(fused_out, out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_model.conv.weight.grad,
                               model.conv.weight.grad)
    torch.testing.assert_close(fused_model.conv.bias.grad,
                               model.conv.bias.grad)

def test_conv_bias_relu():

    class FusedConvBiasAdd(FusedConvBiasAddActivation):

        def forward(self, x, y=None, alpha=1.0):
            conv = self.conv
            return torch.ops.sfast.cudnn_convolution_bias_relu(
                x, conv.weight,  conv.bias, conv.stride, conv.padding,
                conv.dilation, conv.transposed, conv.output_padding,
                conv.groups)

    orig_model = ConvBiasAddActivation(activation_cls=torch.nn.ReLU)
    orig_model.cuda()

    model = copy.deepcopy(orig_model)
    x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    out = aot_printer(model)(x)
    out.sum().backward()

    fused_model = FusedConvBiasAdd(copy.deepcopy(orig_model))
    fused_x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    fused_out = aot_printer(fused_model)(fused_x)
    fused_out.sum().backward()

    torch.testing.assert_close(fused_out, out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_model.conv.weight.grad,
                               model.conv.weight.grad)
    torch.testing.assert_close(fused_model.conv.bias.grad,
                               model.conv.bias.grad)


def test_conv_bias_tanh():

    class FusedConvBiasAdd(FusedConvBiasAddActivation):

        def forward(self, x, y=None, alpha=1.0):
            conv = self.conv
            return torch.ops.sfast.cudnn_convolution_bias_tanh(
                x, conv.weight,  conv.bias, conv.stride, conv.padding,
                conv.dilation, conv.transposed, conv.output_padding,
                conv.groups)

    orig_model = ConvBiasAddActivation(activation_cls=torch.nn.Tanh)
    orig_model.cuda()

    model = copy.deepcopy(orig_model)
    x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    out = aot_printer(model)(x)
    out.sum().backward()

    fused_model = FusedConvBiasAdd(copy.deepcopy(orig_model))
    fused_x = torch.ones(1, 2, 256, 256, requires_grad=True).cuda()
    fused_out = aot_printer(fused_model)(fused_x)
    fused_out.sum().backward()

    torch.testing.assert_close(fused_out, out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(fused_model.conv.weight.grad,
                               model.conv.weight.grad)
    torch.testing.assert_close(fused_model.conv.bias.grad,
                               model.conv.bias.grad)
