'''
This package implements specific metrics in each module, ideally extending the Metric class.
'''
from torch import Tensor


class Metric():
    '''
    All metrics should extend this, and define if lower is better and its name.
    '''
    def __init__(self):
        assert hasattr(self, 'lower_is_best')
        assert hasattr(self, 'name')

    def __call__(self, outputs: Tensor, target: Tensor) -> float:
        raise NotImplementedError
