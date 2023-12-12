import inspect
import functools
import torch
import sfast


class ScriptModuleClearHook:
    def __init__(self, script_module_c):
        self.class_type = sfast._C._jit_get_module_type(script_module_c)

    def __del__(self):
        try:
            sfast._C._jit_clear_class_type_registration(self.class_type)
        except Exception:
            pass


def attach_script_module_clear_hook(
    script_module, attr_name="_sfast_module_registration_clear_hook"
):
    script_module._register_attribute(
        attr_name, torch._C.PyObjectType.get(), ScriptModuleClearHook(script_module)
    )
    for child_name, child_module in torch._C._jit_debug_module_iterators(script_module)[
        "named_children"
    ]:
        attach_script_module_clear_hook(child_module, attr_name)


@functools.wraps(torch.jit.trace)
def better_trace(func, *args, **kwargs):
    script_module = torch.jit.trace(func, *args, **kwargs)
    attach_script_module_clear_hook(script_module._c)
    return script_module


@functools.wraps(torch.jit.freeze)
def better_freeze(script_module, *args, **kwargs):
    if not kwargs.get("preserve_parameters", False):
        torch._C._jit_pass_inline(script_module.graph)
        sfast._C._jit_pass_fix_frozen_conv_folding(script_module.graph)

    freeze = torch.jit.freeze
    if (
        "preserve_parameters" in kwargs
        and "preserve_parameters" not in inspect.signature(freeze).parameters
    ):
        from typing import List, Optional
        from torch.jit._script import RecursiveScriptModule, ScriptModule

        # Based on https://github.com/pytorch/pytorch/blob/7bcf7da3a268b435777fe87c7794c382f444e86d/torch/jit/_freeze.py#L13C1-L13C1
        def freeze(
            mod,
            preserved_attrs: Optional[List[str]] = None,
            optimize_numerics: bool = True,
            preserve_parameters: bool = False,
        ):
            if not isinstance(mod, ScriptModule):
                raise RuntimeError(
                    "Freezing expects a ScriptModule as input. "
                    "Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'."
                )

            if mod.training:
                raise RuntimeError(
                    "Freezing is currently only implemented for modules in eval mode. "
                    "Please call .eval() on your module before freezing."
                )

            preserved_attrs = preserved_attrs if preserved_attrs is not None else []

            out = RecursiveScriptModule(
                torch._C._freeze_module(
                    mod._c, preserved_attrs, preserveParameters=preserve_parameters
                )
            )
            RecursiveScriptModule._finalize_scriptmodule(out)

            preserved_methods = [x for x in preserved_attrs if mod._c._has_method(x)]
            torch.jit.run_frozen_optimizations(
                out, optimize_numerics, preserved_methods
            )

            return out

    freezed_module = freeze(script_module, *args, **kwargs)
    attach_script_module_clear_hook(freezed_module._c)
    return freezed_module
