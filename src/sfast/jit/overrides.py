import torch
from torch.overrides import TorchFunctionMode


class TracingMode(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # https://github.com/pytorch/pytorch/issues/107591
        if func == torch.Tensor.repeat_interleave:
            if len(args) >= 2 and isinstance(args[0],
                                             torch.Tensor) and isinstance(
                                                 args[1], torch.Tensor):
                device = args[0].device
                if device != args[1].device:
                    if args[1].ndim == 0:
                        args = (args[0], args[1].item(), *args[2:])
                    else:
                        args = (args[0], args[1].to(device=device), *args[2:])
        return func(*args, **kwargs)
