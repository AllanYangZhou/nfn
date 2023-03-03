import random
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Dataset
import torch
import pandas as pd


def cycle(loader):
    while True:
        for blah in loader:
            yield blah


class StatTracker:
    def __init__(self):
        self.sum, self.sq_sum, self.count = 0, 0, 0

    def update(self, x: torch.Tensor):
        # x is a tensor of shape (batch_size, ...)
        self.sum += x.sum(dim=0)
        self.sq_sum += (x ** 2).sum(dim=0)
        self.count += x.shape[0]

    def get_mean_std(self):
        mean = self.sum / self.count
        std = torch.sqrt(self.sq_sum / self.count - mean ** 2)
        return mean, std

    
def compute_mean_std(loader, max_batches=None):
    w_trackers, b_trackers = None, None
    i = 0
    for params, *_ in loader:
        if max_batches is not None and i >= max_batches:
            break
        weights, bias = params
        if w_trackers is None:
            w_trackers = [StatTracker() for _ in weights]
            b_trackers = [StatTracker() for _ in bias]
        for w, w_tracker in zip(weights, w_trackers): w_tracker.update(w)
        for b, b_tracker in zip(bias, b_trackers): b_tracker.update(b)
        i += 1
    w_stats = [w_tracker.get_mean_std() for w_tracker in w_trackers]
    b_stats = [b_tracker.get_mean_std() for b_tracker in b_trackers]
    return w_stats, b_stats


class ZooDataset(Dataset):
    def __init__(self, data_path, mode, idcs_file=None):
        data = np.load(os.path.join(data_path, "weights.npy"))
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        metrics = pd.read_csv(os.path.join(data_path, "metrics.csv.gz"), compression='gzip')
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(os.path.join(data_path, "layout.csv"))
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[mode]
        data = data[idcs]
        self.metrics = metrics.iloc[idcs]

        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def __getitem__(self, idx):
        # insert a channel dim
        weights = tuple(w[idx][None] for w in self.weights)
        biases = tuple(b[idx][None] for b in self.biases)
        return (weights, biases), self.metrics.iloc[idx].test_accuracy.item()