import itertools
import functools
import dataclasses
import ctypes
import torch

is_tracing = torch._C._is_tracing


# convert an arbitrary object to a tuple of tensors
def convert_to_flat_tensors(obj):
    return flatten_obj(obj)


# convert a tuple of tensors to an arbitrary object
def convert_from_flat_tensors(tensors):
    # for sfast._C._jit_pass_erase_scalar_tensors
    tensors = tuple(t if isinstance(t, torch.Tensor) else torch.tensor([t])
                    for t in tensors)
    return unflatten_tensors(tensors)[0]


def _tensor_from_int(num):
    return torch.tensor([num], dtype=torch.int64)


_tensor_from_int_cached = functools.lru_cache(maxsize=256)(_tensor_from_int)


def tensor_from_int(num):
    return (_tensor_from_int if is_tracing() else _tensor_from_int_cached)(num)


def flatten_obj(obj):
    if obj is None:
        type_num = 0
        flatten_func = flatten_none
    elif isinstance(obj, torch.Tensor):
        type_num = 1
        flatten_func = flatten_tensor
    # bool must be be checked before int because bool is a subclass of int
    elif isinstance(obj, bool):
        type_num = 2
        flatten_func = flatten_bool
    elif isinstance(obj, float):
        type_num = 3
        flatten_func = flatten_float
    elif isinstance(obj, int):
        type_num = 4
        flatten_func = flatten_int
    elif isinstance(obj, str):
        type_num = 5
        flatten_func = flatten_str
    elif isinstance(obj, bytes):
        type_num = 6
        flatten_func = flatten_bytes
    elif isinstance(obj, list):
        type_num = 7
        flatten_func = flatten_list_or_tuple
    elif isinstance(obj, tuple):
        type_num = 8
        flatten_func = flatten_list_or_tuple
    # dataclass must be checked before dict
    # because dataclass is a subclass of dict
    elif dataclasses.is_dataclass(obj):
        type_num = 9
        flatten_func = flatten_dataclass
    elif isinstance(obj, dict):
        type_num = 10
        flatten_func = flatten_dict
    else:
        type_num = 11
        flatten_func = flatten_unknown

    return (tensor_from_int(type_num), ) + flatten_func(obj)


def flatten_none(obj):
    return tuple()


def flatten_tensor(obj):
    return (obj, )


def _flatten_bool(obj):
    return (torch.tensor([obj], dtype=torch.bool), )


_flatten_bool_cached = functools.lru_cache(maxsize=256)(_flatten_bool)


def flatten_bool(obj):
    return (_flatten_bool if is_tracing() else _flatten_bool_cached)(obj)


def _flatten_float(obj):
    return (torch.tensor([obj], dtype=torch.float64), )


_flatten_float_cached = functools.lru_cache(maxsize=256)(_flatten_float)


def flatten_float(obj):
    return (_flatten_float if is_tracing() else _flatten_float_cached)(obj)


def _flatten_int(obj):
    return (torch.tensor([obj], dtype=torch.int64), )


_flatten_int_cached = functools.lru_cache(maxsize=256)(_flatten_int)


def flatten_int(obj):
    return (_flatten_int if is_tracing() else _flatten_int_cached)(obj)


def flatten_str(obj):
    return flatten_bytes(obj.encode('utf-8'))


def _flatten_bytes(obj):
    return (torch.as_tensor(tuple(obj), dtype=torch.uint8), )


_flatten_bytes_cached = functools.lru_cache(maxsize=256)(_flatten_bytes)


def flatten_bytes(obj):
    return (_flatten_bytes if is_tracing() else _flatten_bytes_cached)(obj)


def flatten_list_or_tuple(obj):
    size = len(obj)
    return (tensor_from_int(size),
            *itertools.chain.from_iterable(flatten_obj(arg) for arg in obj))


def flatten_dict(obj):
    keys = list(obj.keys())
    keys.sort()
    size = len(keys)
    return (tensor_from_int(size), *itertools.chain.from_iterable(
        itertools.chain(flatten_obj(key), flatten_obj(obj[key]))
        for key in keys))


def flatten_dataclass(obj):
    d = dict((field.name, getattr(obj, field.name))
             for field in dataclasses.fields(obj))
    return flatten_unknown(obj.__class__) + flatten_dict(d)


def flatten_unknown(obj):
    return (save_object_reference_in_tensor(obj), )


def unflatten_tensors(tensors, start=0):
    obj_type = tensors[start].item()
    if obj_type == 0:
        return unflatten_none(tensors, start + 1)
    elif obj_type == 1:
        return unflatten_tensor(tensors, start + 1)
    elif obj_type == 2:
        return unflatten_bool(tensors, start + 1)
    elif obj_type == 3:
        return unflatten_float(tensors, start + 1)
    elif obj_type == 4:
        return unflatten_int(tensors, start + 1)
    elif obj_type == 5:
        return unflatten_str(tensors, start + 1)
    elif obj_type == 6:
        return unflatten_bytes(tensors, start + 1)
    elif obj_type == 7:
        return unflatten_list_or_tuple(tensors, start + 1, list)
    elif obj_type == 8:
        return unflatten_list_or_tuple(tensors, start + 1, tuple)
    elif obj_type == 9:
        return unflatten_dataclass(tensors, start + 1)
    elif obj_type == 10:
        return unflatten_dict(tensors, start + 1)
    elif obj_type == 11:
        return unflatten_unknown(tensors, start + 1)
    else:
        raise ValueError("Unknown type number: {}".format(obj_type))


def unflatten_none(tensors, start):
    return None, start


def unflatten_tensor(tensors, start):
    return tensors[start], start + 1


def unflatten_bool(tensors, start):
    return bool(tensors[start].item()), start + 1


def unflatten_float(tensors, start):
    return float(tensors[start].item()), start + 1


def unflatten_int(tensors, start):
    return int(tensors[start].item()), start + 1


def unflatten_str(tensors, start):
    bytes_obj, start = unflatten_bytes(tensors, start)
    return bytes_obj.decode('utf-8'), start


def unflatten_bytes(tensors, start):
    return bytes(tensors[start].tolist()), start + 1


def unflatten_list_or_tuple(tensors, start, list_or_tuple):
    size = tensors[start].item()
    start += 1
    content = []
    for _ in range(size):
        obj, start = unflatten_tensors(tensors, start)
        content.append(obj)
    return list_or_tuple(content), start


def unflatten_dict(tensors, start):
    size = tensors[start].item()
    start += 1
    content = {}
    for _ in range(size):
        key, start = unflatten_tensors(tensors, start)
        value, start = unflatten_tensors(tensors, start)
        content[key] = value
    return content, start


def unflatten_dataclass(tensors, start):
    clz, start = unflatten_unknown(tensors, start)
    content, start = unflatten_dict(tensors, start)
    return clz(**content), start


def unflatten_unknown(tensors, start):
    return restore_object_from_tensor(tensors[start]), start + 1


class ObjectStorationHelper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, obj):
        obj_id = id(obj)
        return torch.tensor([obj_id], dtype=torch.int64)

    @staticmethod
    def backward(ctx, grad):
        return None


save_object_reference_in_tensor = ObjectStorationHelper.apply


class ObjectRestorationHelper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t):
        obj_id = t.item()
        return ctypes.cast(obj_id, ctypes.py_object).value

    @staticmethod
    def backward(ctx, grad):
        return None


restore_object_from_tensor = ObjectRestorationHelper.apply
