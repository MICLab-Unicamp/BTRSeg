'''
Code related to transforming the intensity of data
'''
import random

from transforms import Transform


class RandomIntensity(Transform):
    '''
    Randomly applies intensity transform to the image
    Image should be between 0 and 1
    '''
    def __init__(self, p=1, brightness=0.10, reset_seed=True):
        assert p > 0 and p <= 1 and brightness > 0 and brightness <= 1, "arguments make no sense"
        self.p = p
        self.b = brightness
        self.reset_seed = reset_seed

    def __call__(self, image, mask):
        if self.reset_seed:
            random.seed()
        if random.random() < self.p:
            value = ((random.random() - 0.5)*2)*self.b
            image += value

        return image, mask

    def __str__(self):
        return "Intensity: p {}, brightness {}".format(self.p, self.b)
