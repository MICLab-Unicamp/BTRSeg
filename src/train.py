'''
Experiment Description
BTRSeg architecutre: expansion of the 2D UNet from E2DHipseg to 3D, testing various parameter configurations.
'''
EXPERIMENT_NAME = "BTRSeg"

# Standard Library
import os
import logging
import argparse

# External Libraries
import numpy as np
import torch
from torch.optim import Adam, SGD

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# DLPT
from DLPT.models.unet import UNet
from DLPT.metrics.brats import BraTSMetrics
from DLPT.loss.dice import DICELoss, BalancedMultiChannelDICELoss
from DLPT.optimizers.radam import RAdam
from DLPT.transforms.to_tensor import ToTensor
from DLPT.transforms.patches import ReturnPatch, CenterCrop
from DLPT.transforms.intensity import RandomIntensity
from DLPT.transforms import Compose
from DLPT.datasets.brats import BRATS
from DLPT.utils.git import get_git_hash
from DLPT.utils.reproducible import deterministic_run


BRATS.PATHS["2020"]["default"] = "data/MICCAI_BraTS2020_TrainingData"


class BRATS3DSegmentation(pl.LightningModule):
    def __init__(self, hparams, loss, metric):
        super(BRATS3DSegmentation, self).__init__()
        self.loss_calculator = loss
        self.metric_calculator = metric
        self.hparams = hparams
        self.model = UNet(n_channels=hparams.nin, n_classes=hparams.nout,
                          apply_sigmoid=self.hparams.final_labels, apply_softmax=not(self.hparams.final_labels),
                          residual=True, small=False, bias=False, bn=self.hparams.norm,
                          dim='3d', use_attention=self.hparams.use_attention, channel_factor=self.hparams.channel_factor)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        Training
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_calculator(y_hat, y)  # DICE Loss per channel mean

        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        '''
        Validation step.
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_calculator(y_hat, y)  # DICE Loss per channel mean
        metrics = self.metric_calculator(y_hat, y)
        metrics["loss"] = loss

        return metrics

    def test_step(self, batch, batch_idx):
        '''
        Test. Should be exactly equal to validation step
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = self.loss_calculator(y_hat, y)  # DICE Loss per channel mean
        metrics = self.metric_calculator(y_hat, y)
        metrics["loss"] = loss

        return metrics

    # -----------------------------------------------------------------------------------------------------------------#

    def training_epoch_end(self, outputs):
        '''
        Training
        '''
        name = "train_"

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        return {'log': {name + 'loss': avg_loss}}

    def validation_epoch_end(self, outputs):
        '''
        Validation
        '''
        name = 'val_'

        # Initialize metric dict
        metrics = {}
        for k in outputs[0].keys():
            metrics[name + k] = []

        # Fill metric dict
        for x in outputs:
            for k, v in x.items():
                metrics[name + k].append(v)

        # Get mean for every metric
        for k, v in metrics.items():
            if k == name + 'loss':
                metrics[k] = torch.stack(v).mean()
            else:
                metrics[k] = np.array(v).mean()

        avg_loss = metrics[name + "loss"]
        tqdm_dict = {name + "loss": avg_loss}

        print(tqdm_dict)
        if self.hparams.balanced_dice:
            loss_calculator.increment_weights()

        return {name + 'loss': avg_loss, 'log': metrics, 'progress_bar': tqdm_dict,
                name + "WT_dice": metrics[name + "WT_dice"],
                name + "TC_dice": metrics[name + "TC_dice"],
                name + "ET_dice": metrics[name + "ET_dice"]
                }

    def test_epoch_end(self, outputs):
        '''
        Test
        '''
        name = 'test_'

        # Initialize metric dict
        metrics = {}
        for k in outputs[0].keys():
            metrics[name + k] = []

        # Fill metric dict
        for x in outputs:
            for k, v in x.items():
                metrics[name + k].append(v)

        # Get mean for every metric
        for k, v in metrics.items():
            if k == name + 'loss':
                metrics[k] = torch.stack(v).mean()
            else:
                metrics[k] = np.array(v).mean()

        avg_loss = metrics[name + "loss"]
        tqdm_dict = {name + "loss": avg_loss}

        return {name + 'loss': avg_loss, 'log': metrics, 'progress_bar': tqdm_dict,
                name + "WT_dice": metrics[name + "WT_dice"],
                name + "TC_dice": metrics[name + "TC_dice"],
                name + "ET_dice": metrics[name + "ET_dice"]
                }

    # -----------------------------------------------------------------------------------------------------------------#

    def configure_optimizers(self):
        if self.hparams.opt == "Adam":
            opt = Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.opt == "RAdam":
            opt = RAdam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        elif self.hparams.opt == "SGD":
            opt = SGD(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        return [opt], [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)]

    def train_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="train",
                        transform=transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=True if self.hparams.forced_overfit == 0 else False,
                                      num_workers=0 if self.hparams.cpu else 4)

    def val_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="validation",
                        transform=test_transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.test_bs, shuffle=False, num_workers=0 if self.hparams.cpu else 4)

    def test_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="test",
                        transform=test_transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.test_bs, shuffle=False, num_workers=0 if self.hparams.cpu else 4)


test_transforms = Compose([CenterCrop(128, 128, 128, segmentation=True, assert_big_enough=True),
                           ToTensor(volumetric=True, classify=False)])
model_folder = "models"
log_folder = "mlruns"

if __name__ == "__main__":
    # Logging initialization
    debug = False
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=FORMAT)
    logging.info("Logging initialized with debug={}".format(debug))

    loss_calculator = DICELoss(volumetric=True, per_channel=True)
    metric_calculator = BraTSMetrics(dice_only=True)

    transforms = Compose([ReturnPatch(patch_size=(128, 128, 128), segmentation=True, fullrandom=True,
                                      reset_seed=False), RandomIntensity(reset_seed=False),
                          ToTensor(volumetric=True, classify=False)])
    test_transforms = Compose([CenterCrop(128, 128, 128, segmentation=True, assert_big_enough=True),
                               ToTensor(volumetric=True, classify=False)])

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="desc", type=str)

    # Variables to experimented with
    parser.add_argument("-bs", type=int, required=True, help="Batch size.")
    parser.add_argument("-norm", type=str, required=True, help="batch, norm or none.")
    parser.add_argument("-opt", type=str, required=True, help="Adam or RAdam.")
    parser.add_argument("-precision", type=int, required=True, help="16 (mixed) or 32 (full)")
    parser.add_argument("-balanced_dice", action='store_true', help="Enables the custom balanced dice.")

    # Fixed arguments
    parser.add_argument("-loss", type=str, default=loss_calculator.__class__.__name__)
    parser.add_argument("-metric", type=str, default=metric_calculator.__class__.__name__)
    parser.add_argument("-type", type=str, default="segmentation")
    parser.add_argument("-debug", action="store_true")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-final_labels", type=bool, default=True)
    parser.add_argument("-lr", type=float, default=0.0005)
    parser.add_argument("-wd", type=float, default=1e-5)
    parser.add_argument("-test_bs", type=int, default=1)
    parser.add_argument("-max_epochs", type=int, default=300)
    parser.add_argument("-channel_factor", type=int, default=4)
    parser.add_argument("-nin", type=int, default=4)  # flair, t1, t1ce, t2
    parser.add_argument("-nout", type=int, default=3)  # WT, TC, ET
    parser.add_argument("-transforms", type=str, default=str(transforms))
    parser.add_argument("-test_transforms", type=str, default=str(test_transforms))
    parser.add_argument("-use_attention", action='store_true')
    parser.add_argument("-rseed", type=int, default=4321)
    parser.add_argument("-forced_overfit", type=float, default=0)
    parser.add_argument("-dataset", type=str, default="BRATS")
    parser.add_argument("-dataset_year", type=str, default="2020")
    parser.add_argument("-splits", type=str, default="(0.7, 0.1, 0.2)")
    parser.add_argument("-kfold", type=str, default="None")
    parser.add_argument("-fold", type=str, default="None")

    hyperparameters = parser.parse_args()

    if hyperparameters.balanced_dice:
        print("Setting up BalancedMultiChannelDICELoss")
        loss_calculator = BalancedMultiChannelDICELoss(hyperparameters.max_epochs, True)
        hyperparameters.loss = loss_calculator.__class__.__name__

    print("Experiment Hyperparameters:\n")
    print(vars(hyperparameters))

    # # Initialize Trainer
    # Instantiate model
    model = BRATS3DSegmentation(hyperparameters, loss=loss_calculator, metric=metric_calculator)

    # Folder management
    experiment_name = EXPERIMENT_NAME
    os.makedirs(model_folder, exist_ok=True)
    ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.4f}-{val_WT_dice:.4f}-{val_TC_dice:.4f}-{val_ET_dice:.4f}")

    # Callback initialization
    checkpoint_callback = ModelCheckpoint(prefix=experiment_name + '_' + hyperparameters.desc, filepath=ckpt_path, monitor="val_loss",
                                          mode="min")
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="file:" + log_folder,
                          tags={"desc": hyperparameters.desc, "commit": get_git_hash()})

    # PL Trainer initialization
    trainer = Trainer(gpus=0 if hyperparameters.cpu else 1, precision=hyperparameters.precision, checkpoint_callback=checkpoint_callback,
                      early_stop_callback=False, logger=logger, max_epochs=hyperparameters.max_epochs, deterministic=True,
                      fast_dev_run=hyperparameters.debug, progress_bar_refresh_rate=1, overfit_pct=hyperparameters.forced_overfit)

    # # Training Loop
    seed = hyperparameters.rseed
    print(f"Applying random seed {seed}")
    deterministic_run(seed)
    trainer.fit(model)
