import logging
import torch
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger()

aten = torch.ops.aten

aten_equal_default = aten.equal.default
aten__local_scalar_dense_default = aten._local_scalar_dense.default


def with_dispatch_mode(dispatch_mode):

    def decorator(func):

        def wrapper(*args, **kwargs):
            with dispatch_mode():
                return func(*args, **kwargs)

        return wrapper

    return decorator


class BaseDispatchMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == aten_equal_default:
            args = tuple(torch.as_tensor(x) for x in args)
        elif func == aten__local_scalar_dense_default:
            args = tuple(torch.as_tensor(x) for x in args)
        return func(*args, **kwargs)


class LoggingMode(BaseDispatchMode):

    def __init__(self, logger_=None, level=logging.INFO):
        if logger_ is None:
            logger_ = logger
        self.logger = logger_
        self.level = level
        return super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.logger.log(self.level,
                        f"Calling {func.__module__}.{func.__name__}")
        return super().__torch_dispatch__(func, types, args, kwargs)


class ReplaceFuncMode(BaseDispatchMode):

    def __init__(self, replacements, debug=False):
        self.replacements = replacements
        self.debug = debug
        return super().__init__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        replacement = self.replacements.get(func)
        if replacement is None:
            if self.debug:
                logger.info(f"Calling {func.__module__}.{func.__name__}")
            return super().__torch_dispatch__(func, types, args, kwargs)
        else:
            if self.debug:
                logger.info(
                    f"Replacing {func.__module__}.{func.__name__} with "
                    f"{replacement.__module__}.{replacement.__name__}")
            if kwargs is None:
                kwargs = {}
            return replacement(*args, **kwargs)
