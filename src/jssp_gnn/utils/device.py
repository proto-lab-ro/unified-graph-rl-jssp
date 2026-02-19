import torch


# Default device
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device() -> torch.device:
    """
    Get the globally configured device.

    Returns:
        torch.device: The current device.
    """
    global _device
    return _device


def set_device(device: str | torch.device) -> None:
    """
    Set the global device.

    Args:
        device: The device to set. Can be a string (e.g., "cpu", "cuda", "auto") or a torch.device object.
    """
    global _device
    if isinstance(device, str):
        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)
    else:
        _device = device
