import torch 
import random 
import numpy as np 

def set_seed(seed_value=42):
    """
    Set the random seed for reproducibility in random, numpy, and PyTorch.

    Args:
        seed_value (int): Seed value for random number generators. Default is 42.

    Returns:
        int: The seed value used for setting the random seed.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
    return seed_value