'''
BTRSeg Experiment
Change some parameters by looking at the --help
'''
EXPERIMENT_NAME = "BTRSeg"
FINALIZED = False


# Standard Library
import os
import logging
import argparse
from glob import iglob
from argparse import Namespace

# External Libraries
import numpy as np
import torch

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# DLPT
from DLPT.models.unet import UNet
from DLPT.metrics.brats import BraTSMetrics
from DLPT.loss.dice import DICELoss
from DLPT.optimizers.radam import RAdam
from DLPT.transforms.to_tensor import ToTensor
from DLPT.transforms.patches import ReturnPatch, CenterCrop
from DLPT.transforms.intensity import RandomIntensity
from DLPT.transforms import Compose
from DLPT.datasets.brats import BRATS
from DLPT.utils.git import get_git_hash
from DLPT.utils.reproducible import deterministic_run


class BRATS3DSegmentation(pl.LightningModule):
    def __init__(self, hparams):
        super(BRATS3DSegmentation, self).__init__()

        self.hparams = hparams
        self.model = UNet(n_channels=hparams.nin, n_classes=hparams.nout,
                          apply_sigmoid=self.hparams.final_labels, apply_softmax=not(self.hparams.final_labels),
                          residual=True, small=False, bias=False, bn=self.hparams.norm,
                          dim='3d', use_attention=False, channel_factor=self.hparams.channel_factor)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        Training
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = loss_calculator(y_hat, y)  # DICE Loss per channel mean

        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        '''
        Validation step.
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = loss_calculator(y_hat, y)  # DICE Loss per channel mean
        metrics = metric_calculator(y_hat, y)
        metrics["loss"] = loss

        return metrics

    def test_step(self, batch, batch_idx):
        '''
        Test. Should be exactly equal to validation step
        '''
        x, y, _, _, _ = batch
        y_hat = self.forward(x)
        loss = loss_calculator(y_hat, y)  # DICE Loss per channel mean
        metrics = metric_calculator(y_hat, y)
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
        opt = RAdam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        assert opt.__class__.__name__ == self.hparams.opt
        return [opt], [torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)]

    def train_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="train",
                        transform=transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.bs, shuffle=True, num_workers=0 if args.cpu else 4)

    def val_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="validation",
                        transform=test_transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.test_bs, shuffle=False, num_workers=0 if args.cpu else 4)

    def test_dataloader(self):
        dataset = BRATS(year=self.hparams.dataset_year, release="default", group="all", mode="test",
                        transform=test_transforms, convert_to_eval_format=(self.hparams.nout == 3))
        return dataset.get_dataloader(batch_size=self.hparams.test_bs, shuffle=False, num_workers=0 if args.cpu else 4)


test_transforms = Compose([CenterCrop(128, 128, 128, segmentation=True, assert_big_enough=True),
                           ToTensor(volumetric=True, classify=False)])


if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cpu', action='store_true',
                        help="Uses the CPU. If not present will try to use a GPU.")
    parser.add_argument('-f', '--fast', action='store_true',
                        help="Performs training over a small part of the data, useful for a quick test.")
    parser.add_argument('-e', '--epochs', type=int, default=300, metavar='',
                        help="How long should the training be. Defaults to 300.")
    parser.add_argument('-d', '--data', type=str, default="data/MICCAI_BraTS2020_TrainingData", metavar='',
                        help=("Where the data is, by default its in data/MICCAI_BraTS2020_TrainingData. Make sure you have "
                              "executed the pre-processing."))
    parser.add_argument('-b', '--batch', type=int, default=3, metavar='',
                        help="Batch size. Default 3.")
    parser.add_argument('-p', '--precision', type=str, default="mixed", metavar='',
                        help="Precision, one of 'full' or 'mixed'.")
    parser.add_argument('-ch', '--channel_factor', type=int, default=4, metavar='',
                        help="The higher this value, the smaller the resulting network. Default is 4.")
    parser.add_argument('-t', '--tag', type=str, required=True,
                        help="A description of this run.")
    args = parser.parse_args()

    # Logging initialization
    debug = False
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format=FORMAT)
    logging.info("Logging initialized with debug={}".format(debug))

    model_folder = "models"
    log_folder = "mlruns"

    segmentation = True
    loss_calculator = DICELoss(volumetric=True, per_channel=True)
    metric_calculator = BraTSMetrics(dice_only=True)
    opt_name = "RAdam"

    forced_overfit = False
    channel_factor = args.channel_factor
    nin = 4  # flair, t1, t1ce, t2
    nout = 3  # WT, TC, EC
    architecture = EXPERIMENT_NAME

    transforms = Compose([ReturnPatch(patch_size=(128, 128, 128), segmentation=segmentation, fullrandom=True,
                                      reset_seed=False), RandomIntensity(reset_seed=False),
                          ToTensor(volumetric=True, classify=not segmentation)])

    tags = {"desc": args.tag, "commit": get_git_hash()}

    hyperparameters = {"experiment_name": EXPERIMENT_NAME, "type": "segmentation", "achitecture": architecture,
                       "with_aug": True, "final_labels": True, "lr": 0.0005, "wd": 1e-5, "bs": args.batch, "test_bs": 1,
                       "max_epochs": args.epochs,
                       "channel_factor": channel_factor, "nin": nin, "nout": nout,
                       "transforms": str(transforms), "test_transforms": str(test_transforms), "use_attention": False,
                       "norm": "group", "rseed": 4321, "opt": opt_name, "loss": loss_calculator.name, "desc": tags["desc"],
                       "metric": metric_calculator.name,
                       "forced_overfit": 0.05 if args.fast else 0,
                       "precision": 16 if args.precision == "mixed" else 32,
                       "dataset": "BRATS", "dataset_year": "2020", "splits": "(0.7, 0.1, 0.2)",
                       "kfold": "None", "fold": "None"}

    # Sets folder according to parameters.
    assert os.path.isdir(args.data), "Data source doesn't exist! Have you performed downloading/unpack/preprocessing steps?"
    assert len(list(iglob(os.path.join(args.data, "**", "*.npz"), recursive=True))) == 369, ("there should be 369 npzs in "
                                                                                             "data folder. Something is "
                                                                                             "wrong with data.")
    BRATS.PATHS[hyperparameters["dataset_year"]]["default"] = args.data

    print("Experiment Hyperparameters:\n")
    for key, parameter in hyperparameters.items():
        print("{}: {}".format(key, parameter))

    # # Initialize Trainer
    # Instantiate model
    model = BRATS3DSegmentation(Namespace(**hyperparameters))

    # Folder management
    experiment_name = hyperparameters["experiment_name"]
    os.makedirs(model_folder, exist_ok=True)
    ckpt_path = os.path.join(model_folder, "-{epoch}-{val_loss:.4f}-{val_WT_dice:.4f}-{val_TC_dice:.4f}-{val_EC_dice:.4f}")

    # Callback initialization
    checkpoint_callback = ModelCheckpoint(prefix=experiment_name + '_' + tags["desc"], filepath=ckpt_path, monitor="val_loss",
                                          mode="min")
    logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri="file:" + log_folder,
                          tags=tags)

    # PL Trainer initialization
    trainer = Trainer(gpus=0 if args.cpu else 1, precision=hyperparameters["precision"],
                      checkpoint_callback=checkpoint_callback, early_stop_callback=False, logger=logger,
                      max_epochs=hyperparameters["max_epochs"], fast_dev_run=False, progress_bar_refresh_rate=1,
                      overfit_pct=hyperparameters["forced_overfit"])

    # Training Loop
    if not FINALIZED:  # Change to train when you want to train and
        seed = hyperparameters["rseed"]
        print(f"Applying random seed {seed}")
        deterministic_run(seed)
        trainer.fit(model)
    else:
        print("This training is finalized. Are you sure you running the correct file?")
