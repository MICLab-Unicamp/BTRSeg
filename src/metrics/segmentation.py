'''
Contains implementations related to DICE
'''
import sys
import torch
import logging
from torch import Tensor, einsum
from typing import List

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from metrics import Metric
from utils.asserts import simplex


class DICEMetric(Metric):
    '''
    Calculates DICE Metric
    '''
    def __init__(self, apply_sigmoid=False, mask_ths=0.5, skip_ths=False, per_channel_metric=False):
        self.name = "DICE"
        self.lower_is_best = False
        super(DICEMetric, self).__init__()
        self.apply_sigmoid = apply_sigmoid
        self.mask_ths = mask_ths
        self.skip_ths = skip_ths
        self.per_channel_metric = per_channel_metric
        print(f"DICE Metric initialized with apply_sigmoid={apply_sigmoid}, mask_ths={mask_ths}, skip_ths={skip_ths}, "
              f"per_channel={per_channel_metric}")

    def __call__(self, probs, target):
        '''
        Returns only DICE metric, as volumetric dice
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: float binary target mask
        '''
        probs = probs.type(torch.float32)
        target = target.type(torch.float32)

        if self.apply_sigmoid:
            probs = probs.sigmoid()

        p_min = probs.min()
        assert p_min >= 0.0, "FATAL ERROR: DICE metric input not positive! Did you apply sigmoid?"

        if self.skip_ths:
            mask = probs
        else:
            mask = (probs > self.mask_ths).float()

        if self.per_channel_metric:
            assert len(target.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                            "volumes")
            nchannels = target.shape[1]
            return [vol_dice(mask[:, c], target[:, c], smooth=0.0).item() for c in range(nchannels)]
        else:
            return vol_dice(mask, target, smooth=0.0).item()


def vol_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of volume
    '''
    # q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum()
    tflat_sum = tflat.sum()

    if iflat_sum.item() == 0.0 and tflat_sum.item() == 0.0:
        # print("DICE Metric got black mask and prediction!")
        dice = torch.tensor(1.0, requires_grad=True, device=inpt.device)
    else:
        dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


def batch_dice(inpt, target, smooth=1.0):
    '''
    Calculate DICE of a batch of two binary masks
    Returns mean dice of all slices
    '''
    q = inpt.size(0)
    assert len(inpt) != 0, " trying to compute DICE of nothing"

    iflat = inpt.contiguous().view(q, -1)
    tflat = target.contiguous().view(q, -1)
    intersection = (iflat * tflat).sum(dim=1)

    eps = 0
    if smooth == 0.0:
        eps = sys.float_info.epsilon

    iflat_sum = iflat.sum(dim=1)
    tflat_sum = tflat.sum(dim=1)

    dice = (2. * intersection + smooth) / (iflat_sum + tflat_sum + smooth + eps)

    dice = dice.mean()
    value = dice.item()
    assert value >= 0.0 or value <= 1.0, " DICE not between 0 and 1! something is wrong"

    return dice


class GeneralizedDice():
    '''
    Code from Boundary loss for highly unbalanced segmentation
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.cross_entropy = False
        self.loss = kwargs["loss"]
        self.name = "GD" + 'L'*self.loss
        self.lower_is_best = self.loss
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        if self.loss:
            divided = 1 - divided

        loss = divided.mean()

        return loss


class Generalized3DDice():
    '''
    Code from Boundary loss for highly unbalanced segmentation
    '''
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.cross_entropy = False
        self.loss = kwargs["loss"]
        self.name = "GD" + 'L'*self.loss
        self.lower_is_best = self.loss
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcdwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcdwh,bcdwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcdwh->bc", pc) + einsum("bcdwh->bc", tc))

        divided: Tensor = 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        if self.loss:
            divided = 1 - divided

        loss = divided.mean()

        return loss


# These are deprecated in favor of using SKLearn
def precision(pred, tgt):
    '''
    True positives / (true positives + false positives)
    '''
    assert pred.shape == tgt.shape
    if isinstance(pred, np.ndarray):
        ones = np.ones_like(pred)
    else:
        ones = torch.ones_like(pred)

    one_minus_tgt = ones - tgt

    TP = (pred*tgt).sum()  # Postiv
    FP = (pred*one_minus_tgt).sum()  # Negatives that are in prediction

    TPplusFP = TP + FP
    if TPplusFP == 0:
        logging.warning("Can't measure precision without positives")
        return torch.zeros(1)
    else:
        return TP/TPplusFP


def recall(pred, tgt):
    return sensitivity(pred, tgt)


def sensitivity(pred, tgt):
    '''
    True positive rate, how many positives are actually positive
    Supports torch or numpy
    '''
    tgt_sum = tgt.sum()
    if tgt_sum == 0:
        logging.warning("Can't measure sensitivity without positive targets")
        return torch.zeros(1)
    else:
        return (pred*tgt).sum() / tgt.sum()


def specificity(pred, tgt):
    '''
    True negative rate, how many negatives are actually negative
    Doesnt work well with too many true negatives
    '''
    assert pred.shape == tgt.shape
    if isinstance(pred, np.ndarray):
        ones = np.ones_like(pred)
    else:
        ones = torch.ones_like(pred)

    ones_minus_tgt = ones - tgt
    ones_minus_pred = ones - pred

    one_minus_tgt_sum = ones_minus_tgt.sum()

    if one_minus_tgt_sum == 0:
        logging.warning("Can't measure specificity without negative targets")
        return torch.zeros(1)

    return ((ones_minus_pred)*(ones_minus_tgt)).sum() / one_minus_tgt_sum


def numpy_haussdorf_95(pred: np.ndarray, target: np.ndarray) -> float:
    '''
    95th percentile version inspired by:
    "Automatic lung segmentation in routine imaging is a
    data diversity problem, not a methodology problem"
    '''
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max((np.percentile(directed_hausdorff(pred, target)[0], 95),
                np.percentile(directed_hausdorff(target, pred)[0], 95)))
#
