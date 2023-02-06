import glob
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.utils.data as data
from src.models.core import parse_t_f
from tqdm import tqdm


class GraphDataset(data.Dataset):
    def __init__(
        self,
        dataset_dir,
        n_fourier=5,
        n_nodes_in_sample=1000,
        time=False,
        cache_fourier=True,
        in_memory=True,
        cut=-1,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.n_fourier = n_fourier
        self.n_nodes_in_sample = n_nodes_in_sample
        self.time = time
        self.cache_fourier = cache_fourier
        self._fourier = None
        self._fourier_path = os.path.join(dataset_dir, "fourier.npy")
        self.in_memory = in_memory
        self.cut = cut

        self.filenames = self.get_filenames(dataset_dir)
        if cut > 0:
            self.filenames = self.filenames[:cut]
        self.npzs = [np.load(f) for f in self.filenames]
        self._data = None
        if in_memory:
            print("Loading dataset")
            self._data = [self.load_data(i) for i in tqdm(range(len(self)))]

    def load_data(self, index):
        data = {}

        data["inputs"] = self.get_inputs(index)
        data["target"] = self.get_target(index)

        return data

    def get_fourier(self, index):
        if self.cache_fourier and os.path.exists(self._fourier_path):
            if self._fourier is None:
                self._fourier = np.load(self._fourier_path)
                self._fourier = torch.from_numpy(self._fourier).float()
                self._fourier = self._fourier[:, : self.n_fourier]
            return self._fourier
        else:
            arr = self.npzs[index]["fourier"][:, : self.n_fourier]
            return torch.from_numpy(arr).float()

    def get_time(self, index):
        arr = self.npzs[index]["time"]
        return torch.from_numpy(arr).float()

    @staticmethod
    def add_time(points, time):
        n_points = points.shape[-2]
        time = time.unsqueeze(0).repeat(n_points, 1)
        return torch.cat([points, time], dim=-1)

    def get_inputs(self, index):
        arr = self.get_fourier(index)
        if self.time:
            time = self.get_time(index)
            arr = self.add_time(arr, time)

        return arr

    def get_target(self, index):
        arr = self.npzs[index]["target"]
        return torch.from_numpy(arr).float()

    def get_data(self, index):
        if self.in_memory:
            return self._data[index]
        else:
            return self.load_data(index)

    def __getitem__(self, index):
        data = self.get_data(index)
        data_out = dict()

        n_points = data["inputs"].shape[0]
        points_idx = self.get_subsampling_idx(n_points, self.n_nodes_in_sample)
        data_out["inputs"] = data["inputs"][points_idx]
        data_out["target"] = data["target"][points_idx]
        data_out["index"] = index

        return data_out

    def __len__(self):
        return len(self.filenames)

    @property
    def target_dim(self):
        return self.get_data(0)["target"].shape[-1]

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--dataset_dir", default="data/", type=str)
        parser.add_argument("--n_fourier", default=5, type=int)
        parser.add_argument("--n_nodes_in_sample", default=1000, type=int)
        parser.add_argument("--time", type=parse_t_f, default=False)
        parser.add_argument("--in_memory", type=parse_t_f, default=True)
        parser.add_argument("--cut", default=-1, type=int)

        return parser

    @staticmethod
    def get_filenames(dataset_dir, subset=None):
        if subset is None:
            subset = ["*"]

        if isinstance(subset, str):
            subset = open(subset).read().splitlines()
        elif isinstance(subset, list):
            pass
        else:
            raise TypeError(
                f"Unsupported type {type(subset)} for subset. "
                f"Expected string or list."
            )

        npz_dir = os.path.join(dataset_dir, "npz_files")
        npz_filenames = []
        for f in subset:
            npz_filenames += glob.glob(os.path.join(npz_dir, f"{f}.npz"))

        npz_filenames = sorted(npz_filenames, key=lambda s: s.split("/")[-1])

        return npz_filenames

    @staticmethod
    def get_subsampling_idx(n_points, to_keep):
        if n_points >= to_keep:
            idx = torch.randperm(n_points)[:to_keep]
        else:
            # Sample some indices more than once
            idx = (
                torch.randperm(n_points * int(np.ceil(to_keep / n_points)))[:to_keep]
                % n_points
            )

        return idx
