import dataclasses
import torch
import sfast


def tree_copy_(dest, src):
    if isinstance(dest, torch.Tensor):
        dest.copy_(src)
    elif isinstance(dest, (list, tuple)):
        assert len(dest) == len(src)
        for x, y in zip(dest, src):
            tree_copy_(x, y)
    elif dataclasses.is_dataclass(dest):
        assert len(dest) == len(src)
        for field in dataclasses.fields(dest):
            tree_copy_(getattr(dest, field.name), getattr(src, field.name))
    elif isinstance(dest, dict):
        assert len(dest) == len(src)
        for k in dest:
            tree_copy_(dest[k], src[k])
    else:
        assert type(dest) is type(src)


def tree_copy(src, detach=False):
    if isinstance(src, torch.Tensor):
        return src.detach().clone() if detach else src.clone()
    elif isinstance(src, (list, tuple)):
        return type(src)(tree_copy(x, detach=detach) for x in src)
    elif dataclasses.is_dataclass(src):
        return type(src)(
            **{
                field.name: tree_copy(getattr(src, field.name), detach=detach)
                for field in dataclasses.fields(src)
            })
    elif isinstance(src, dict):
        return type(src)(
            (k, tree_copy(v, detach=detach)) for k, v in src.items())
    else:
        return src


def shadow_copy(obj, detach=False):
    if isinstance(obj, torch.Tensor):
        return sfast._C._create_shadow_tensor(
            obj, detach=detach) if obj.device.type == 'cuda' else obj
    elif isinstance(obj, (list, tuple)):
        return type(obj)(shadow_copy(x, detach=detach) for x in obj)
    elif dataclasses.is_dataclass(obj):
        return type(obj)(**{
            field.name:
            shadow_copy(getattr(obj, field.name), detach=detach)
            for field in dataclasses.fields(obj)
        })
    elif isinstance(obj, dict):
        return type(obj)(
            (k, shadow_copy(v, detach=detach)) for k, v in obj.items())
    else:
        return obj


def can_be_perfectly_copied(obj):
    if obj is None:
        return True
    elif isinstance(obj, torch.Tensor):
        return True
    # micro optimization: bool obj is an instance of int
    elif isinstance(obj, (str, int, float, bytes)):
        return True
    elif isinstance(obj, (list, tuple)):
        return all(can_be_perfectly_copied(x) for x in obj)
    elif dataclasses.is_dataclass(obj):
        return all(
            can_be_perfectly_copied(getattr(obj, field.name))
            for field in dataclasses.fields(obj))
    elif isinstance(obj, dict):
        return all(can_be_perfectly_copied(v) for v in obj.values())
    else:
        return False
