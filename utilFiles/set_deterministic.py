import random
import numpy as np
import torch

def make_deterministic(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    torch.set_printoptions(precision=5)