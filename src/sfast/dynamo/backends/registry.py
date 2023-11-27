import functools


@functools.lru_cache(None)
def _lazy_import():
    from .. import backends
    from torch._dynamo.utils import import_submodule

    import_submodule(backends)