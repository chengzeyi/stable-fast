import sys

if sys.version_info < (3, 8):
    # Fix for Triton < 2.0.1 bug
    # TypeError: Expected maxsize to be an integer or None
    import functools

    lru_cache = functools.lru_cache

    def new_lru_cache(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return lru_cache()(args[0])
        else:
            return lru_cache(*args, **kwargs)

    functools.lru_cache = new_lru_cache

from .utils.env import setup_environment

setup_environment()

try:
    import sfast._C as _C
except ImportError:
    print('''
***ERROR IMPORTING sfast._C***
Unable to load stable-fast C extension.
Is it compatible with your PyTorch installation?
Or is it compatible with your CUDA version?
''')
    raise

# This line will be programatically read/write by setup.py.
# Leave them at the bottom of this file and don't touch them.
__version__ = "0.0.14"
