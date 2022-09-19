import functools
import operator
import random
from typing import List

import numpy as np
import torch


def determinist_behavior(seed: int = 1) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def warn(msg: str, **kwargs) -> None:
    print(color.BOLD + color.YELLOW + msg + color.END, **kwargs)


def info(msg: str, **kwargs) -> None:
    print(color.BOLD + color.GREEN + msg + color.END, **kwargs)


def error(msg: str, **kwargs) -> None:
    print(color.BOLD + color.RED + msg + color.END, **kwargs)


def unravel(l: List) -> List:
    "[(0, 1), (2, 3), (4, 5)] -> [0, 1, 2, 3, 4, 5]"
    return functools.reduce(operator.iconcat, l, [])


def fix_layer_names(checkpoint) -> None:
    """Fix layer names in a checkpoint if trained in dataparallel mode."""
    # {k.replace("module.", ""): v for k, v in checkpoint["model"].items()}
    for k in list(checkpoint["model"]):
        if "module." in k:
            checkpoint["model"][k.replace(
                "module.", "")] = checkpoint["model"][k]
