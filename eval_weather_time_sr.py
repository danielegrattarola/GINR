"""
Evaluation script for time interpolation in the weather modeling experiment.
Evaluates the model on time_factor * 24 time steps equispaced in [0, 1], using 
a pre-computed Fourier basis of the spherical mesh obtained by subdividing the 
training mesh with PyMesh.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file
    - time_factor: predict the signal for time_factor * 24 time steps equispaced
                   in [0, 1]. Setting time_factor=1 is equivalent to evaluating
                   on the training set.
    - cmap: color map to use when creating the plots/gif
    - append: string to append to the output folder (for identification)
    - predict: by default, the script will look for a npy with the predicted
               signals and avoids predicting again if it finds one.
               This flag overrides the behavior, forcing the prediction regardless.
    - no_plots: do not generate any plots.

IMPORTANT NOTE: you must pass the flag --time=True for this script to work
Note: requires the --dataset_dir flag to be specified as well.
"""

import os
from argparse import ArgumentParser

import imageio
import numpy as np
import pytorch_lightning as pl
import torch
from plotly import graph_objects as go
from tqdm import tqdm

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.utils.data_generation import cartesian_to_sphere
from src.utils.get_predictions import get_batched_predictions

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--time_factor", default=2, type=int)  # Training: 1
parser.add_argument("--cmap", default="Spectral", type=str)
parser.add_argument("--append", default="", type=str)
parser.add_argument("--predict", action="store_true")
parser.add_argument("--no_plots", action="store_true")
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

# Data
dataset = GraphDataset(**vars(args))
u = np.load(os.path.join(dataset.dataset_dir, "fourier_sr_aligned_cut.npy"))
points = np.load(os.path.join(dataset.dataset_dir, "points.npy"))
x, y, z = points.T
lats, lons = cartesian_to_sphere(x, y, z)

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)


def get_pred_at(t):
    """Utility to predict the signal at a given time instant t"""
    inputs = torch.from_numpy(u).float()[:, : args.n_fourier]
    inputs = dataset.add_time(inputs, torch.from_numpy(np.array(t)).float())
    _, pred = get_batched_predictions(model, inputs, 0, verbose=0)
    return pred


# Plot parameters
output_dir = f"output_weather_time_gifs_tf={args.time_factor}_sr_{args.append}/"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
w = h = 2048 * 2
time_factor = args.time_factor  # Increase the time resolution by this much

# Predict signals for t = [0, ..., 1] and save them to disk
test_times = np.arange(0, 24 * time_factor) / (24 * time_factor)
pred_signals_path = os.path.join(output_dir, "output_weather_time_pred_signals.npy")
if os.path.exists(pred_signals_path) and not args.predict:
    pred_signals = np.load(pred_signals_path)
else:
    pred_signals = np.array([get_pred_at(t) for t in tqdm(test_times)])
    np.save(pred_signals_path, pred_signals)

# Plots
if not args.no_plots:
    for i, signal in enumerate(tqdm(pred_signals)):
        fig = go.Figure(
            go.Scattergeo(
                lat=lats.reshape(-1),
                lon=lons.reshape(-1),
                mode="markers",
                marker_symbol="square",
                marker_line_width=0,
                marker_color=signal.reshape(-1),
                marker_opacity=0.5,
                marker_colorscale=args.cmap,
                marker_cmax=pred_signals.max(),
                marker_cmin=pred_signals.min(),
            )
        )
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
        img_path = os.path.join(
            output_dir, "imgs", f"pred_{np.round(i / time_factor, 2):.2f}.png"
        )
        fig.write_image(img_path, width=w, height=h)
        os.system(f"convert -trim {img_path} {img_path}")

    images = []
    for i in range(len(pred_signals)):
        images.append(
            imageio.imread(
                os.path.join(
                    output_dir, "imgs", f"pred_{np.round(i / time_factor, 2):.2f}.png"
                )
            )
        )
    gif_path = os.path.join(output_dir, f"pred.gif")
    imageio.mimsave(gif_path, images, fps=len(images))
    os.system(f"gifsicle -O3 {gif_path} -o {gif_path}")
    imageio.mimsave(os.path.join(output_dir, f"pred.mp4"), images, fps=len(images))
