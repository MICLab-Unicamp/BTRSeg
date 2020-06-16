'''
Handles random things being always in the same order
'''
import random
import numpy as np
import torch


def deterministic_run(seed=4321):
    '''
    Source: https://pytorch.org/docs/stable/notes/randomness.html
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
