import copy
import types
import functools


def copy_func(f, globals=None, module=None, name=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    if name is None:
        name = f.__name__
    g = types.FunctionType(f.__code__,
                           globals,
                           name=name,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    if module is not None:
        g.__module__ = module
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    g.__name__ = name
    return g
