import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DEFAULTS = {
    "seed": 42,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def get_device():
    return DEFAULTS["device"]
