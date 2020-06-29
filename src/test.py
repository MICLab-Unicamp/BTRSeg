import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from DLPT.metrics.brats import BraTSMetrics
from DLPT.loss.dice import DICELoss
from train import BRATS3DSegmentation, log_folder

print("Testing...")

with torch.set_grad_enabled(False):
    cpu = not torch.cuda.is_available()
    if cpu:
        print("CUDA enabled GPU not found, falling back to CPU testing.")

    logger = MLFlowLogger(experiment_name="best_model_testing", tracking_uri="file:" + log_folder,)

    best_model = BRATS3DSegmentation.load_from_checkpoint("models/best_model.ckpt",
                                                          loss=DICELoss(volumetric=True, per_channel=True),
                                                          metric=BraTSMetrics(dice_only=True))
    best_model.hparams.cpu = cpu

    tester = Trainer(gpus=1 - int(cpu), logger=logger)
    tester.test(best_model)
