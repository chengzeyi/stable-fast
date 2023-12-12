import logging
import threading
from sfast.utils.patch import patch_module

logger = logging.getLogger()


def apply_to_all_modules(m, compiler, filter_func=None):
    if filter_func is None:
        filter_func = lambda stack: True
    return patch_module(m, filter_func, lambda m: apply_to_module(m, compiler))


def apply_to_module(m, compiler):
    ModuleJitHook(m, compiler)
    return m


class ModuleJitHook:

    def __init__(self, module, compiler):
        self.lock = threading.Lock()
        self.module = module
        self.compiler = compiler
        self.compiled_cache = {}

        self.call_impl = self.module._call_impl
        self.module._call_impl = self.compiled_call_impl

    def compiled_call_impl(self, *args, **kwargs):
        if self.compiler.is_compiling():
            return self.call_impl(*args, **kwargs)
        inputs_key = self.compiler.get_inputs_key(self.call_impl, args, kwargs)
        if inputs_key is None:
            return self.call_impl(*args, **kwargs)
        compiled = self.compiled_cache.get(inputs_key)
        if compiled not in (None, self.ready_to_compile, self.cannot_compile):
            return compiled(*args, **kwargs)
        with self.lock:
            if inputs_key in self.compiled_cache:
                compiled = self.compiled_cache[inputs_key]
                if compiled == self.ready_to_compile:
                    logger.info(f"Compiling {self.module.__class__.__name__}")
                    compiled = self.compiler.compile(self.call_impl, args,
                                                     kwargs)
                    self.compiled_cache[inputs_key] = compiled
                elif compiled == self.cannot_compile:
                    return self.call_impl(*args, **kwargs)
                return compiled(*args, **kwargs)
            outputs = self.call_impl(*args, **kwargs)
            outputs_key = self.compiler.get_outputs_key(
                self.call_impl, outputs)
            if outputs_key is None:
                self.compiled_cache[inputs_key] = self.cannot_compile
            else:
                self.compiled_cache[inputs_key] = self.ready_to_compile
            return outputs

    def ready_to_compile(self):
        pass

    def cannot_compile(self):
        pass
