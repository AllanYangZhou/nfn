import typing
import glob
import re
import random
import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Dataset, ConcatDataset, Subset, Sampler
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import pandas as pd

from nfn.common import state_dict_to_tensors


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


class SirenDataset(Dataset):
    def __init__(
        self,
        data_path,
        prefix="randinit_test",
        split="all",
        # split point for val and test sets
        split_points: typing.Tuple[int, int] = None,
    ):
        idx_pattern = r"net(\d+)\.pth"
        label_pattern = r"_(\d)s"
        self.idx_to_path = {}
        self.idx_to_label = {}
        # TODO: this glob pattern should actually be f"{prefix}_[0-9]s/*.pth".
        # For 1 original + 10 augs, this amounts to having 10 copies instead of 11,
        # so it probably doesn't make a big difference in final performance.
        for siren_path in glob.glob(os.path.join(data_path, f"{prefix}_*/*.pth")):
            idx = int(re.search(idx_pattern, siren_path).group(1))
            self.idx_to_path[idx] = siren_path
            label = int(re.search(label_pattern, siren_path).group(1))
            self.idx_to_label[idx] = label
        if split == "all":
            self.idcs = list(range(len(self.idx_to_path)))
        else:
            val_point, test_point = split_points
            self.idcs = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(self.idx_to_path))),
            }[split]

    def __getitem__(self, idx):
        data_idx = self.idcs[idx]
        sd = torch.load(self.idx_to_path[data_idx])
        weights, biases = state_dict_to_tensors(sd)
        return (weights, biases), self.idx_to_label[data_idx]

    def __len__(self):
        return len(self.idcs)


DEF_TFM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))])
class SirenAndOriginalDataset(Dataset):
    def __init__(
        self,
        siren_path,
        siren_prefix,
        data_path,
        split="all",
        data_tfm=DEF_TFM,
        split_points=None,
    ):
        siren_dset = SirenDataset(siren_path, split="all", prefix=siren_prefix)
        if "mnist" in siren_path:
            self.data_type = "mnist"
            print("Loading MNIST")
            MNIST_train = MNIST(data_path, transform=data_tfm, train=True, download=True)
            MNIST_test = MNIST(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([MNIST_train, MNIST_test])
        elif "fashion" in siren_path:
            self.data_type = "fashion"
            print("Loading FashionMNIST")
            fMNIST_train = FashionMNIST(data_path, transform=data_tfm, train=True, download=True)
            fMNIST_test = FashionMNIST(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([fMNIST_train, fMNIST_test])
        else:
            self.data_type = "cifar"
            print("Loading CIFAR10")
            CIFAR_train = CIFAR10(data_path, transform=data_tfm, train=True, download=True)
            CIFAR_test = CIFAR10(data_path, transform=data_tfm, train=False, download=True)
            dset = ConcatDataset([CIFAR_train, CIFAR_test])
        if split == "all":
            idcs = list(range(len(siren_dset)))
        else:
            val_point, test_point = split_points
            idcs = {
                "train": list(range(val_point)),
                "val": list(range(val_point, test_point)),
                "test": list(range(test_point, len(siren_dset))),
            }[split]
        self.siren_dset = Subset(siren_dset, idcs)
        self.dset = Subset(dset, idcs)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        params, siren_label = self.siren_dset[idx]
        img, data_label = self.dset[idx]
        assert siren_label == data_label
        return params, img, data_label


class AlignedSampler(Sampler):
    def __init__(self, data_source, samples_per_dset):
        """Assumes data_source is a concatenation of N datasets, each of size samples_per_dset."""
        self.data_source = data_source
        assert len(self.data_source) % samples_per_dset == 0, "Dataset size must be divisible by samples_per_dset."
        self.num_dsets = len(self.data_source) // samples_per_dset
        self.samples_per_dset = samples_per_dset

    def __iter__(self):
        example_indices = torch.randperm(self.samples_per_dset).tolist()
        # get the example_indices[i]th example from each dataset, for i in range(samples_per_dset)
        full_indices = []
        for idx in example_indices:
            full_indices.extend([idx + i * self.samples_per_dset for i in range(self.num_dsets)])
        yield from full_indices

    def __len__(self):
        return len(self.data_source)