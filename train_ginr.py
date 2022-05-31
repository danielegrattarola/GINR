import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from implicit_graphs.data.graph_dataset import GraphDataset
from implicit_graphs.models.graph_inr import GraphINR
from implicit_graphs.plotting.figures import draw_pc
from implicit_graphs.utils.get_predictions import get_batched_predictions

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument("--patience", default=5000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_workers", default=0, type=int)
parser.add_argument("--plot_3d", action="store_true")
parser.add_argument("--plot_heat", action="store_true")
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

# Data
dataset = GraphDataset(**vars(args))
loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
)

# Model
input_dim = dataset.n_fourier + (1 if dataset.time else 0)
output_dim = dataset.target_dim
model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))

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
    max_epochs=-1,
    log_every_n_steps=1,
    callbacks=[checkpoint_cb, earlystopping_cb, lrmonitor_cb],
    logger=logger,
    gpus=torch.cuda.device_count(),
    strategy="ddp" if torch.cuda.device_count() > 1 else None,
)
trainer.fit(model, loader)
model.load_from_checkpoint(checkpoint_cb.best_model_path)

try:
    points = dataset.npzs[0]["points"]
except KeyError:
    points = np.load(os.path.join(dataset.dataset_dir, "points.npy"))

inputs = dataset.get_inputs(0)
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_pc(points, pred[:, 0], colorscale="Reds")
fig.show()
logger.experiment.log({"Scatter": fig})
