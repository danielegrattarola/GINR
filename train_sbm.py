import itertools
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import scipy.sparse as sp
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.graph_inr import GraphINR
from src.utils.data_generation import normalized_laplacian
from src.utils.eigenvectors import align_eigenvectors_kl
from src.utils.get_predictions import get_batched_predictions


def make_dataset(groups, inter=0.1, intra=0.5, n_per_group=500, k=100):
    sizes = [n_per_group] * groups
    ids = [[i] * s for i, s in enumerate(sizes)]
    ids = np.array([i for g in ids for i in g])
    p = (np.ones((groups, groups)) - np.eye(groups)) * inter
    p += np.eye(groups) * intra
    g = nx.stochastic_block_model(sizes, p)
    a = nx.adjacency_matrix(g)
    a = a.astype("f4")
    l = normalized_laplacian(a)
    v, u = sp.linalg.eigsh(l, k=k, which="SM")

    return u, ids


pl.seed_everything(1234)

parser = ArgumentParser()

parser.add_argument("--patience", default=5000, type=int)
parser.add_argument("--n_workers", default=0, type=int)
parser.add_argument("--n_fourier", default=100, type=int)
parser.add_argument("--groups", default=2, type=int)
parser.add_argument("--inter", default=0.1, type=float)
parser.add_argument("--intra", default=0.5, type=float)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
args = parser.parse_args()

# Data
n_per_group = 500
u, y = make_dataset(
    args.groups,
    inter=args.inter,
    intra=args.intra,
    k=args.n_fourier,
    n_per_group=n_per_group,
)
u, y = torch.from_numpy(u), torch.from_numpy(y)


class CustomDataset(Dataset):
    def __getitem__(self, item):
        return {"inputs": u, "target": y, "index": 0}

    def __len__(self):
        return 1


dataset = CustomDataset()
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.n_workers)

# Model
input_dim = args.n_fourier
output_dim = args.groups
model = GraphINR(input_dim, output_dim, len(dataset), classifier=True, **vars(args))

# Training
checkpoint_cb = ModelCheckpoint(
    monitor="loss", mode="min", save_last=True, filename="best"
)
earlystopping_cb = EarlyStopping(monitor="loss", patience=args.patience)
lrmonitor_cb = LearningRateMonitor(logging_interval="step")
logger = WandbLogger(project="GINR", save_dir="lightning_logs")
logger.experiment.log(
    {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", None)}
)
trainer = pl.Trainer.from_argparse_args(
    args,
    log_every_n_steps=1,
    callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb],
    logger=logger,
    gpus=torch.cuda.device_count(),
    strategy="ddp" if torch.cuda.device_count() > 1 else None,
)
trainer.fit(model, loader)
model.load_from_checkpoint(checkpoint_cb.best_model_path)


# Test on different graphs
inter_range = np.linspace(0.1, 1.0, 100)
intra_range = np.linspace(1.0, 0.1, 100)
scores = []
repetitions = 1
for i, (inter, intra) in enumerate(tqdm(itertools.product(inter_range, intra_range))):
    nmi = acc = 0
    for _ in range(repetitions):
        u_test, y_test = make_dataset(
            args.groups, inter=inter, intra=intra, k=args.n_fourier
        )
        u_test = align_eigenvectors_kl(u.numpy(), u_test)
        u_test = torch.from_numpy(u_test)
        _, pred = get_batched_predictions(model, u_test, 0)
        pred = pred.argmax(-1)
        nmi += normalized_mutual_info_score(y_test, pred)
        acc += accuracy_score(y_test, pred)
    scores.append([inter, intra, nmi / repetitions, acc / repetitions])

# Plot results
scores = np.array(scores).reshape(inter_range.shape[0], intra_range.shape[0], -1)
np.save("plots/sbm_scores.npy", scores)
plt.figure(figsize=(3.3, 3.3))
plt.pcolormesh(
    scores[..., 0], scores[..., 1], scores[..., 2], cmap="RdBu", rasterized=True
)
plt.gca().set_aspect("equal")
n = n_per_group * 2
threshold = (np.log(n) / n) * ((np.sqrt(2) + np.sqrt(inter_range * n / np.log(n))) ** 2)
mask = threshold <= 1
plt.plot(inter_range[mask], threshold[mask], color="C1")
plt.ylabel("$r$ (intra)")
plt.xlabel("$p$ (inter)")
for tick in plt.gca().xaxis.get_major_ticks():
    tick.label.set_rotation(45)
    tick.set_pad(0)
plt.colorbar(fraction=0.04575, pad=0.04)
plt.scatter(0.1, 0.5, marker="x", color="C1", clip_on=False)
plt.tight_layout()
plt.savefig(f"sbm_{args.groups}_groups.pdf", bbox_inches="tight")
plt.show()
