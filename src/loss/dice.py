'''
Using DICE as a Loss
'''
import torch

from metrics.segmentation import vol_dice, batch_dice

from loss import Loss


class DICELoss(Loss):
    '''
    Calculates DICE Loss
    Dont use with multiple targets. Use GDL.
    '''
    def __init__(self, volumetric=False, negative_loss=False, per_channel=False):
        self.name = "DICE Loss"
        super(DICELoss, self).__init__()
        self.volumetric = volumetric
        self.negative_loss = negative_loss
        self.per_channel = per_channel

        print(f"DICE Loss initialized with volumetric={volumetric}, negative? {negative_loss}, per_channel {per_channel}")

    def __call__(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE loss input not bounded! Did you apply sigmoid?"

        score = 0

        if self.per_channel:
            assert len(targets.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                             "volumes")
            nchannels = targets.shape[1]
            if self.volumetric:
                score = torch.stack([vol_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
            else:
                score = torch.stack([batch_dice(probs[:, c], targets[:, c]) for c in range(nchannels)]).mean()
        else:
            if self.volumetric:
                score = vol_dice(probs, targets)
            else:
                score = batch_dice(probs, targets)

        if self.negative_loss:
            loss = -score
        else:
            loss = 1 - score

        return loss


class BalancedMultiChannelDICELoss(Loss):
    '''
    Specific case of DICE loss with multichannel softmax input instead of sigmoid
    Balance works by multiplying the loss value by the MSE of DICE scores per channel, multiplied by alpha.
    '''
    def __init__(self, volumetric=False, alpha=1):
        '''
        volumetric: true if input Tensors will be 5D (B, C, X, Y, Z) False if 4D (B, C, X, Y)
        alpha: parameter to control the weight of balancing factor.
        '''
        self.name = "DICE Loss"
        super(DICELoss, self).__init__()
        self.volumetric = volumetric
        self.alpha = alpha

        print(f"Balanced MultiChannel DICELoss initialized with volumetric={volumetric}")

    def __call__(self, probs, targets):
        '''
        probs: output of last convolution, sigmoided or not (use apply_sigmoid=True if not)
        targets: binary target mask
        '''
        p_min = probs.min()
        p_max = probs.max()
        assert p_max <= 1.0 and p_min >= 0.0, "FATAL ERROR: DICE loss input not bounded! Did you apply sigmoid?"

        assert len(targets.shape) >= 4, ("less than 4 dimensions makes no sense with multi channel in a batch of 2D or 3D"
                                         "volumes")
        nchannels = targets.shape[1]
        if self.volumetric:
            dices = torch.stack([vol_dice(probs[:, c], targets[:, c]) for c in range(nchannels)])
        else:
            dices = torch.stack([batch_dice(probs[:, c], targets[:, c]) for c in range(nchannels)])

        # Inneficient implementation, however size of dices is [B, C], small.
        dists = torch.zeros_like(dices)
        for nb, sample_scores in enumerate(dices):
            for channel_score in sample_scores:
                dists[nb] += (sample_scores - channel_score).abs()

        loss = 1 + self.alpha*dists.mean() - dices.mean()  # optimal results mean lower distances and higher dice (scores)

        return loss
