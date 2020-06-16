'''
Code related to transforming types to the PyTorch Tensor type
'''
import torch
from transforms import Transform


class ToTensor(Transform):
    '''
    Converts numpy arrays to pytorch tensors
    '''
    def __init__(self, volumetric, classify=False):
        self.volumetric = volumetric
        self.classify = classify

    def __call__(self, npimage, npmask):
        '''
        input numpy image: H x W
        output torch image: C X H X W
        '''
        if npimage.ndim == 2 or (npimage.ndim == 3 and self.volumetric):
            image = torch.unsqueeze(torch.from_numpy(npimage), 0).float()
        else:
            image = torch.from_numpy(npimage).float()

        if self.classify:
            mask = torch.from_numpy(npmask).long()
        else:
            if npmask.ndim == 2 or (npmask.ndim == 3 and self.volumetric):
                mask = torch.unsqueeze(torch.from_numpy(npmask), 0).float()
            else:
                mask = torch.from_numpy(npmask).float()

        return image, mask

    def __str__(self):
        return "ToTensor: volumetric {}, classify: {}".format(self.volumetric, self.classify)
