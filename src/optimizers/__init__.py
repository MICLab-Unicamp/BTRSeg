'''
This package implements PyTorch optimizers
'''
from torch import optim
from optimizers.radam import RAdam


def get_optimizer(name, params, lr):
    if name == "RAdam":
        return RAdam(params, lr=lr)
    elif name == "Adam":
        return optim.Adam(params, lr=lr)
    elif name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError("Invalid optimizer name")
