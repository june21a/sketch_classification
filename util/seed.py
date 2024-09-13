import random
import os
import numpy as np
import torch


def seed_everything(seed: int):
    """시드 고정

    Args:
        seed (int): seed number
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id : int):
    """dataloader seed 고정

    Args:
        worker_id (int): seed number
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
