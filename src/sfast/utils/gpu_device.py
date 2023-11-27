import torch


def device_has_tensor_core():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major >= 7
    return False
