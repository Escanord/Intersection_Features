import torch
import numpy as np
import random

def set_seeds_all(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)