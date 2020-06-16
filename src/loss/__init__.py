'''
This package has modules implementing specific loss functions, ideally extending the main Loss class.
'''
from torch import Tensor


class Loss():
    '''
    Every DLPT Loss should extend this
    '''
    def __init__(self):
        self.lower_is_best = True
        assert(hasattr(self, "name"))

    def __call__(self, outputs: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
