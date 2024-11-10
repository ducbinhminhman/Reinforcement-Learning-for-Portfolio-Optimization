import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

def set_global_seed(seed):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)
