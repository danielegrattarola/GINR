"""
Evaluation script for the reaction-diffusion experiment with the Stanford bunny.
Loads the training mesh from the data_generation folder.
Evaluates the model on 3000 time steps equispaced in [0, 1].

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file
    - idx: integer index of a sample from the dataset to predict and plotting
    - sample_every: instead of predicting on the entire 3000 time steps, predict
                    one every [sample_every] time step
    - gif_sample_every: same as above, but only for generating the output gif
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
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.plotting.figures import draw_mesh
from src.utils.data_generation import load_mesh
from src.utils.get_predictions import get_batched_predictions

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser.add_argument("--idx", default=0, type=int)
parser.add_argument("--sample_every", default=1, type=int)  # Training: 10
parser.add_argument("--gif_sample_every", default=5, type=int)  # Training: 10
parser.add_argument("--predict", action="store_true")
parser.add_argument("--no_plots", action="store_true")
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

# Data
dataset = GraphDataset(**vars(args))
mesh_train = load_mesh("data_generation/bunny/reconstruction/bun_zipper.ply")

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)


def get_pred_at(t):
    """Utility to predict the signal at a given time instant t"""
    inputs = dataset.get_fourier(0)
    inputs = dataset.add_time(inputs, torch.from_numpy(np.array(t)).float())
    _, pred = get_batched_predictions(model, inputs, 0, verbose=0)
    return pred


# Plot parameters
output_dir = "output_time_gifs/"
os.makedirs(output_dir, exist_ok=True)
scale = 1 / 8
w = h = int(1024 / scale)
sample_every = args.sample_every

# Predict signals for t = [0, ..., 1] and save them to disk
test_times = np.arange(0, 3000, sample_every) / 3000  # Includes all train_times
pred_signals_path = os.path.join(output_dir, "pred_signal.npy")
if os.path.exists(pred_signals_path) and not args.predict:
    pred_signals = np.load(pred_signals_path)
else:
    pred_signals = np.array([get_pred_at(t) for t in tqdm(test_times)])
    np.save(pred_signals_path, pred_signals)

# Compute R2 score between predicted signal and ground truth
scores = []
for t in test_times:
    idx = int(t * 3000)
    if idx % 10 != 0:
        pred = pred_signals[idx]
        true = np.load(f"dataset/bunny_time_full/npz_files/bunny_time_{idx}.npz")[
            "target"
        ]
        score = r2_score(true, pred)
        scores.append(score)
scores = np.array(scores)
print(f"Test R2: {scores.mean()} +- {scores.std()}")

# Plots
if not args.no_plots:
    inputs = dataset.get_inputs(args.idx)
    target = dataset.get_target(args.idx)
    _, pred = get_batched_predictions(model, inputs, args.idx, verbose=0)

    rot = R.from_euler("xyz", [90, 00, 145], degrees=True).as_matrix()
    fig = draw_mesh(mesh_train, intensity=pred, colorscale="Reds", rot=rot)
    fig.show()

    for i, signal in enumerate(tqdm(pred_signals[:: args.gif_sample_every])):
        fig = draw_mesh(
            mesh_train,
            intensity=signal,
            colorscale="Reds",
            rot=rot,
        )
        fig.update_traces(showlegend=False, showscale=False)
        fig.update_layout(scene_camera=dict(eye=dict(x=1.2, y=1.2, z=0.2)))
        fig.write_image(
            os.path.join(output_dir, f"pred_{i * sample_every}.png"),
            width=w,
            height=h,
            scale=scale,
        )

    images = []
    for i in range(len(pred_signals) // args.gif_sample_every):
        images.append(
            imageio.imread(os.path.join(output_dir, f"pred_{i * sample_every}.png"))
        )
    imageio.mimsave(os.path.join(output_dir, f"pred.gif"), images, fps=60)
    imageio.mimsave(os.path.join(output_dir, f"pred.mp4"), images, fps=60)
