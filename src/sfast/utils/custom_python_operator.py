import torch
import sfast

registered_custom_python_operator_names = set()


def register_custom_python_operator(schema, callable):
    name = torch._C.parse_schema(schema).name
    if name in registered_custom_python_operator_names:
        return
    sfast._C._jit_register_custom_python_operator(schema, callable)
    registered_custom_python_operator_names.add(name)
