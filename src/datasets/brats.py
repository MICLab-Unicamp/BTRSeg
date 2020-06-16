'''
Dealing with the BratS dataset
'''
import logging
import os
import torch
from glob import glob, iglob
from typing import Sequence

import numpy as np

from datasets import Dataset
# from visualization.itksnap import ITKManager


class BRATS(Dataset):
    '''
    Dataset abstraction to handle pre-processed .npz files from BraTS data.
    Expects npzs with at least the "data" key.
    Channel order for data should follow MODALITY_ORDER and channel order for target should follow ORIGINAL_TARGETS.

    parameters:
        year: one of 2018, 2019, 2020, to use pre_loaded paths.
        release: what brats release is being used (time-gated validation and test releases)
        group: select LGG or HGG tumors (not supported in 2020 yet)
        mode: one of train, validation, test, random shuffle with fixed seed.
        transform: data transform
        meta_only: if True, avoids loading the 4D data (usually for meta_data statistics)
        duplicate_lgg: duplicates LGG subjects for classification balance
        sep_mode: 'holdout' or 'kfold'
        splits: (train, val, test) split percentages
        kfold: how many folds to use in kfold
        fold: which fold division to return
        regression_only: removes subjects that dont have survival regression information
    '''
    ORIGINAL_TARGETS = ['bg', 'edema', 'nonenhancing', 'enhancing']
    EVAL_CLASSES = ["WT", "TC", "ET"]
    CLASSES = ["LGG", "HGG"]
    RESECTION = ["nan", "STR", "GTR"]
    MODALITY_ORDER = ["flair", "t1", "t1ce", "t2"]
    PATHS = {"2018": {"default": "/home/diedre/Dropbox/bigdata/brats/2018/train",
                      "test_2": "/home/diedre/Dropbox/bigdata/brats/2018/validation"},
             "2019": {"default": "/home/diedre/Dropbox/bigdata/brats/2019/train",
                      "test_2": "/home/diedre/Dropbox/bigdata/brats/2019/validation"},
             "2020": {"default": "/home/diedre/Dropbox/bigdata/brats/2020/MICCAI_BraTS2020_TrainingData",
                      "val_release": '26 Jun',
                      "test_release": '17 Aug'}}

    def __init__(self, year, release, group, mode, transform=None, meta_only=False, duplicate_lgg=False,
                 sep_mode="holdout", splits=(0.7, 0.1, 0.2), kfold=None, fold=None,
                 convert_to_eval_format=False, has_survival_only=False, verbose=False):
        super(BRATS, self).__init__()
        assert year in ["all", "2018", "2019", "2020"]
        assert group in ["all"] + BRATS.CLASSES
        assert release in ["default"], "only default supported so far"
        assert mode in ["all", "train", "validation", "test"]

        self.year = year
        self.duplicate_lgg = duplicate_lgg
        self.release = release
        self.group = group
        self.mode = mode
        self.transform = transform
        self.meta_only = meta_only
        self.convert_to_eval_format = convert_to_eval_format
        self.has_survival_only = has_survival_only  # keep only those with survival information
        self.verbose = verbose

        try:
            path = BRATS.PATHS[self.year][release]
        except IndexError:
            print("Pre saved release not found, attempting to use path as a path.")
            path = release

        assert os.path.isdir(path), "Given path is not a directory."

        # All Subjects or select group
        if self.group == "all":
            self.subjects = sorted(list(iglob(os.path.join(path, "**", "*.npz"), recursive=True)))
        else:
            if year == "2020":
                raise NotImplementedError("Group selection not working for 2020.")
            self.subjects = sorted(list(iglob(os.path.join(path, self.group, "**", "*.npz"), recursive=True)))

        if has_survival_only:
            # Work around to work with any possible folder
            survival_list = self.has_survival_list(path)

            to_remove = []
            for subject in self.subjects:
                if os.path.basename(subject) not in survival_list:
                    to_remove.append(subject)
            # Never modify what you are iterating.
            for to_r in to_remove:
                self.subjects.remove(to_r)

        # Duplicate LGG subjects for classification balancing
        if duplicate_lgg:
            if year == "2020":
                raise NotImplementedError("Duplicate LGG not working for 2020.")
            logging.warning("DUPLICATING LGGs")
            self.subjects += list(glob(os.path.join(path, "LGG", "**", "*.npz"), recursive=True))

        # Fold/mode Selection
        if self.mode != "all":
            self.subjects = self.get_holdout_or_kfold_set(self.subjects, mode, sep_mode=sep_mode, splits=splits,
                                                          kfold=kfold, fold=fold)

        assert len(self.subjects) > 0, "Something is wrong with subjects list. Did you give the correct path?"

        logging.info(f"BRATS initialized with year: {self.year}, release: {self.release},  group: {self.group}, "
                     f"mode: {self.mode}, convert_to_eval_format: {convert_to_eval_format}, "
                     f"has_survival_only: {has_survival_only}, verbose: {verbose}, "
                     f"meta_only: {self.meta_only}, duplicate_lgg: {self.duplicate_lgg}, transform: {self.transform}, "
                     f"sep_mode: {sep_mode}, splits: {splits}, kfold: {kfold}, fold: {fold}, "
                     f"nsubjects: {len(self.subjects)}.\n")

    def __len__(self) -> int:
        return len(self.subjects)

    @staticmethod
    def original_to_eval_format(y):
        '''
        Transforms a torch in original data format to brats evaluation format (check class variables).
        '''
        try:
            device = y.device
        except AttributeError:
            TORCH = False
        else:
            TORCH = True

        if len(y.shape) == 4:
            mode = "channel_first"
            if TORCH:
                adjusted_y = torch.zeros(size=(3,) + y.shape[1:], dtype=y.dtype, device=device)
            else:
                adjusted_y = np.zeros(shape=(3,) + y.shape[1:], dtype=y.dtype)
        elif len(y.shape) == 5:
            mode = "batch_first"
            if TORCH:
                adjusted_y = torch.zeros(size=(y.shape[0], 3) + y.shape[2:], dtype=y.dtype, device=device)
            else:
                adjusted_y = np.zeros(shape=(y.shape[0], 3) + y.shape[2:], dtype=y.dtype)
        else:
            raise ValueError("Unsupported y shape in brats conversion to eval format.")

        for c, clas in enumerate(BRATS.EVAL_CLASSES):
            if mode == "batch_first":
                if clas == "WT":
                    adjusted_y[:, c] = y[:, 1] + y[:, 2] + y[:, 3]
                elif clas == "TC":
                    adjusted_y[:, c] = y[:, 2] + y[:, 3]
                elif clas == "ET":
                    adjusted_y[:, c] = y[:, 3]
            elif mode == "channel_first":
                if clas == "WT":
                    adjusted_y[c] = y[1] + y[2] + y[3]
                elif clas == "TC":
                    adjusted_y[c] = y[2] + y[3]
                elif clas == "ET":
                    adjusted_y[c] = y[3]

        return adjusted_y

    def get_meta_data(self, npz):
        meta_data = []
        for key in ["tumor_type", "age", "survival"]:
            if key in npz and npz[key] != 'unk':
                if key == "tumor_type":
                    meta_data.append(BRATS.CLASSES.index(str(npz["tumor_type"])))
                else:
                    try:
                        add = float(npz[key].item())
                    except ValueError:
                        if key == "age":
                            raise ValueError("Problem with ages.")
                        add = 360
                    meta_data.append(add)
            else:
                meta_data.append(-1)

        assert np.array([not isinstance(x, str) for x in meta_data]).all(), str(meta_data)

        return tuple(meta_data)

    def __getitem__(self, x: int) -> Sequence:
        if self.verbose:
            print(self.subjects[x])

        npz = np.load(self.subjects[x])

        if self.meta_only:
            vol = tgt = -1
        else:
            vol = npz["data"]
            if "target" in npz:
                tgt = npz["target"].astype(np.float)

        meta_data = self.get_meta_data(npz)

        if self.transform is not None:
            vol, tgt = self.transform(vol, tgt)

        if self.convert_to_eval_format:
            tgt = BRATS.original_to_eval_format(tgt)

        return (vol, tgt) + meta_data  # TODO this change breaks tests, fix

    def has_survival_list(self, path):
        '''
        Remove all files that dont have Survival information.
        '''
        assert self.year == "2020", "has survival only doesnt work with year != 2020"
        print("Reading survival only subjects from has_survival_2020_train.txt")
        try:
            with open(os.path.join(path, "has_survival_2020_train.txt"), 'r') as survival_file:
                return eval(survival_file.read())
        except FileNotFoundError:
            print("Didn't found has_survival_2020_train.txt file in data folder.")
            quit()


def test_brats(display=False, long_test=False):
    from collections import Counter
    from visualization.multiview import MultiViewer, brats_preparation

    for year in ["2020", "2019", "2018"]:
        for mode in ["train", "validation", "test"]:
            brats = BRATS(year, "default", "all", mode, sep_mode="holdout", meta_only=False)
            if long_test:
                brats = BRATS(year, "default", "all", mode, sep_mode="holdout", meta_only=True)
                tgts = brats.get_all_tgts(tgt_axis=2)
                count = Counter(tgts)
                balancing = np.array(tgts).mean()
                logging.info("year: {}, Mode: {}, count: {}, balancing: {}".format(year, mode, count, balancing))

            if display:
                brats = BRATS(year, "default", "all", mode, sep_mode="holdout",
                              has_survival_only=True if year == "2020" else False,
                              convert_to_eval_format=True, transform=None)
                brats = iter(brats)
                quit_signal = False
                while not quit_signal:
                    try:
                        data, target, tt, age, survival = next(brats)
                        print(f"tumor type: {tt}, age: {age}, survival: {survival}")
                        fake_npz = {"data": data, "target": target}
                        data, target = brats_preparation(fake_npz)
                        quit_signal = MultiViewer(data, mask=target).display(channel_select=0) == "ESC"
                    except StopIteration:
                        break

                if quit_signal:
                    return
