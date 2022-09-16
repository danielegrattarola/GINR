"""
Evaluates a trained model by predicting on one sample from the specified dataset.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file
    - idx: integer index of a sample from the dataset to predict and plotting
    - mesh: if not None, a path to a mesh file (obj, ply, etc.) on which to plotting the
            predicted signal. If None, the signal is plotted as a scatter plotting.
    - batch_size: number of points per batch when predicting with the trained model
                  (higher is better)

Note: requires the --dataset_dir flag to be specified as well.
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
from plotly import graph_objects as go

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.plotting import traces
from src.plotting.figures import PLOT_CONFIGS, draw_mesh
from src.utils.data_generation import load_mesh
from src.utils.get_predictions import get_batched_predictions

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--idx", default=0, type=int)
parser.add_argument("--mesh", default=None, type=str)
parser.add_argument("--key", type=str, default="bunny")
parser.add_argument("--batch_size", default=1000000, type=int)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

assert args.key in [
    "bunny",
    "protein_1AA7_A",
], '--key must be in ["bunny", "protein_1AA7_A"]'

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)

# Data
dataset = GraphDataset(**vars(args))
points = dataset.npzs[0]["points"]
inputs = dataset.get_inputs(0)

# Predict
_, pred = get_batched_predictions(model, inputs, 0, batch_size=args.batch_size)

if args.mesh is not None:
    mesh = load_mesh(args.mesh)
    key = args.key
    rot = PLOT_CONFIGS[key]["rot"]
    fig = draw_mesh(
        mesh,
        intensity=pred,
        rot=rot,
        colorscale=PLOT_CONFIGS[key]["colorscale"],
        lower_camera=PLOT_CONFIGS[key]["lower_camera"],
    )
    fig.show()
    fig.write_html(f"{args.key}.html")
else:
    fig = go.Figure(traces.scatter_3d(points, pred))
    fig.show()
