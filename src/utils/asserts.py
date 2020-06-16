'''
Various assertion functions, useful in many modules.
'''
import os
import logging
from typing import Set, Iterable

import torch
from torch import Tensor


VALID_MODES = ["train", "validation", "test"]
orientations = ["sagital", "coronal", "axial"]


def type_assert(_type, *args):
    '''
    Asserts all args to be of type
    '''
    for arg in args:
        assert isinstance(arg, _type)


def assert_orientation(orientation: str):
    '''
    Asserts if given orientation is valid.
    '''
    assert orientation in orientations


def assert_mode(mode: str):
    '''
    Asserts if given move is valid.
    '''
    assert mode in VALID_MODES


def assert_noformat_path(path, warn_only=False) -> str:
    '''
    Checks if path has format, returns path without format or raises assertionerror
    '''
    split_path = os.path.splitext(path)

    if warn_only:
        if len(split_path) > 1:
            logging.warning("No need to add file format to the results save path.")
            return split_path[0]
    else:
        assert len(split_path) > 1
        return path


# Assert utils from github -> LIVIAETS/surface-loss
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, atol=1e-03)


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
#
