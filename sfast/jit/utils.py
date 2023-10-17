import torch
import sfast
import functools


class ScriptModuleClearHook:

    def __init__(self, script_module_c):
        self.class_type = sfast._C._jit_get_module_type(script_module_c)

    def __del__(self):
        sfast._C._jit_clear_class_type_registration(self.class_type)


def attach_script_module_clear_hook(script_module,
                                    attr_name='_module_registration_clear_hook'
                                    ):
    script_module._register_attribute(
        attr_name, torch._C.PyObjectType.get(),
        ScriptModuleClearHook(script_module))
    for child_name, child_module in torch._C._jit_debug_module_iterators(
            script_module)['named_children']:
        attach_script_module_clear_hook(child_module, attr_name)


@functools.wraps(torch.jit.trace)
def better_trace(func, *args, **kwargs):
    script_module = torch.jit.trace(func, *args, **kwargs)
    attach_script_module_clear_hook(script_module._c)
    return script_module


@functools.wraps(torch.jit.freeze)
def better_freeze(script_module, *args, **kwargs):
    freezed_module = torch.jit.freeze(script_module, *args, **kwargs)
    attach_script_module_clear_hook(freezed_module._c)
    return freezed_module
