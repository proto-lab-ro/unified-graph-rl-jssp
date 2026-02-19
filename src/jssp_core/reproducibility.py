import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across various libraries.

    This function sets seeds for:
    - random
    - numpy
    - torch (CPU and CUDA)

    It also configures torch for deterministic execution if requested.

    Args:
        seed: The integer seed to use.
        deterministic: If True, sets torch backends to deterministic mode.
                       This may impact performance but ensures reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True is important because some operations might not have deterministic implementations
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # Fallback for older torch versions
            pass
        os.environ["PYTHONHASHSEED"] = str(seed)
