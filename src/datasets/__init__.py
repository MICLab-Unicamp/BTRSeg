'''
This package contains implementations of specific datasets, each with a dataloader.
'''
import random
from multiprocessing import cpu_count
from typing import Sequence, List
from math import isclose

import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

from utils.asserts import assert_mode


class Dataset(torch.utils.data.Dataset):
    '''
    Extendes PyTorch dataset, including common functions used in datasets.
    '''
    def __init__(self):
        super(Dataset, self).__init__()
        self.target_only = None  # should be replaced by child class argument

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, x: int) -> Sequence:
        raise NotImplementedError

    def get_dataloader(self, batch_size: int, shuffle: bool, num_workers: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    def get_holdout_or_kfold_set(self,
                                 keys: List[str],
                                 mode: str,
                                 sep_mode: str,
                                 splits: float = Sequence[float],
                                 kfold: int = None,
                                 fold: int = None,
                                 shuffle_seed: int = 4321) -> List[str]:
        '''
        keys: list of keys (usually paths) to be divided
        sep_mode: one of 'holdout' or 'kfold'
        splits: sequence indicating how many keys are added to each set, eg: (0.7, 0.1, 0.2) means 70% train, 10% val and
        20% test. Only used for holdout sep_mode.
        kfold: how many folds to create
        fold: which fold to return
        kfold_validation_split: how much of the train set to be separated for validation
        shuffle_seed: seed to always shuffle input keys in the same way. Don't change this if you want consistent folds.

        returns:
            train, validation and test dictionary.
        '''
        assert_mode(mode)
        if sep_mode == "holdout":
            assert isclose(np.array(splits).sum(), 1.0)
        elif sep_mode == "kfold":
            assert isinstance(kfold, int) and isinstance(fold, int) and fold <= kfold and fold > 0 and kfold > 1
        else:
            raise ValueError("Unsupported sep_mode, should be one of 'holdout' or 'kfold'")

        random.seed(shuffle_seed)
        random.shuffle(keys)
        if sep_mode == "holdout":
            full_len = len(keys)
            train_limit = int(full_len*splits[0])
            val_limit = train_limit + int(full_len*splits[1])
            train_set = keys[:train_limit]
            val_set = keys[train_limit:val_limit]
            test_set = keys[val_limit:]
        elif sep_mode == "kfold":
            kfold_generator = KFold(n_splits=kfold)
            train_idx, test_idx = list(kfold_generator.split(keys))[fold - 1]

            train_val_set = [keys[tri] for tri in train_idx]
            train_len = int(len(train_val_set)*(1 - splits[1]))

            train_set = train_val_set[:train_len]
            val_set = train_val_set[train_len:]
            test_set = [keys[tei] for tei in test_idx]

        if mode == "train":
            return train_set
        elif mode == "validation":
            return val_set
        elif mode == "test":
            return test_set

    def get_all_tgts(self, tgt_axis=1):
        '''
        Returns all tgt labels
        '''
        ncpu = cpu_count()
        dataloader = self.get_dataloader(batch_size=len(self)//ncpu, shuffle=False, num_workers=ncpu)

        tgts = torch.cat([batch[tgt_axis] for batch in tqdm(dataloader)]).numpy().tolist()

        return tgts
