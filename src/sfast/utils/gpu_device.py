import torch


def device_has_tensor_core():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major >= 7
    return False


def device_has_capability(major, minor):
    if torch.cuda.is_available():
        major_, minor_ = torch.cuda.get_device_capability()
        return (major_, minor_) >= (major, minor)
    return False
