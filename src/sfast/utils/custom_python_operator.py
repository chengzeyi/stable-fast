import logging
import torch
import sfast

logger = logging.getLogger()

registered_custom_python_operator_names = set()


def register_custom_python_operator(schema, callable):
    name = torch._C.parse_schema(schema).name
    if name in registered_custom_python_operator_names:
        return

    def wrapper(*args, **kwargs):
        try:
            return callable(*args, **kwargs)
        except Exception:
            logger.exception(
                'Exception raised from custom Python operator implementation.')
            raise

    sfast._C._jit_register_custom_python_operator(schema, wrapper)
    registered_custom_python_operator_names.add(name)
