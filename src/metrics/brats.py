'''
Metrics for the BraTS challenge.

"Consistent with the configuration of previous BraTS challenges, we use the "Dice score", and the "Hausdorff distance (95%)".
Expanding upon this evaluation scheme, since BraTS'17 we also use the metrics of "Sensitivity" and "Specificity",
allowing to determine potential over- or under-segmentations of the tumor sub-regions by participating methods."

Note that Sensitivity = Recall
'''
import numpy as np
import torch
from metrics import Metric
from datasets.brats import BRATS
from metrics.segmentation import DICEMetric, specificity, numpy_haussdorf_95
from sklearn.metrics import recall_score


class BraTSMetrics(Metric):
    '''
    This will output metrics for Whole tumor, Tumor core and enchancing core.
    Will deal with cases the network output is 4 class softmax or sigmoid
    directly to the final 3 classes.
    '''
    def __init__(self, dice_only=False):
        self.name = "BraTSMetrics"
        self.lower_is_best = False
        super().__init__()

        self.dice = DICEMetric(per_channel_metric=True)
        if not dice_only:
            self.haussdorf = average_haussdorf_95_over_5Ddata
            self.sensitivity = recall_score
            self.specificity = specificity

        self.dice_only = dice_only

    def __call__(self, y_hat, y):
        '''
        returns dict with [WT, TC, ET] -> []_dice, haussdorf, sensitivity, specificity
        '''
        assert (len(y.shape) == 5) and (y.shape == y_hat.shape), f"{y.shape} {y_hat.shape}"
        assert y_hat.max() <= 1 and y_hat.min() >= 0, "Predictions are not bounded between 0 and 1"
        with torch.set_grad_enabled(False):
            report = {}

            C = y.shape[1]

            # Already in final format
            if C == 3:
                adjusted_y = y
                adjusted_y_hat = y_hat
            elif C == 4:  # need adjustments
                adjusted_y = BRATS.original_to_eval_format(y)
                adjusted_y_hat = BRATS.original_to_eval_format(y_hat)

            # Works over torch
            dices = self.dice(adjusted_y_hat, adjusted_y)

            if not self.dice_only:
                # Other metrics require numpy
                adjusted_y = adjusted_y.cpu().numpy()
                adjusted_y_hat = adjusted_y_hat.detach().cpu().numpy()

                haussdorfs = self.haussdorf(adjusted_y_hat, adjusted_y)

            for i, c in enumerate(BRATS.EVAL_CLASSES):
                report[f"{c}_dice"] = dices[i]

                if not self.dice_only:
                    report[f"{c}_haussdorf"] = haussdorfs[i]

                    npy = (adjusted_y[:, i] > 0.5).astype(np.int).flatten()
                    npy_hat = (adjusted_y_hat[:, i] > 0.5).astype(np.int).flatten()

                    report[f"{c}_sensitivity"] = self.sensitivity(npy, npy_hat, average='binary')
                    report[f"{c}_specificity"] = self.specificity(npy_hat, npy)

            return report


def average_haussdorf_95_over_5Ddata(preds: np.ndarray, target: np.ndarray) -> float:
    '''
    Averages haussdorf 95 over all slices of all channels of all batches of the input 5 dimensional data.
    '''
    assert preds.shape == target.shape

    B, C, _, _, S = preds.shape

    binary_softmax = C == 2

    if binary_softmax:
        haussdorfs = np.zeros((B, S))
    else:
        haussdorfs = np.zeros((B, C, S))

    if binary_softmax:
        for b in range(B):
            for s in range(S):
                haussdorfs[b, s] = numpy_haussdorf_95(preds[b, 0, :, :, s], target[b, 0, :, :, s])
    else:
        for b in range(B):
            for c in range(C):
                for s in range(S):
                    haussdorfs[b, c, s] = numpy_haussdorf_95(preds[b, c, :, :, s], target[b, c, :, :, s])

    haussdorfs = haussdorfs.mean(axis=0).mean(axis=-1)  # reduce batch and slice dimension. Can still return Channels if present.

    return haussdorfs


def test_brats_metrics(display=False, long_test=False):
    import torch

    y = (torch.rand((2, 3, 128, 128, 128)) > 0.5).long()
    y_hat = torch.rand((2, 3, 128, 128, 128))
    brats_metrics = BraTSMetrics()(y_hat, y)

    print(brats_metrics)
