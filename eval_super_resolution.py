"""
Evaluation script for the super-resolution experiment with the Stanford bunny.
Loads the training mesh from the data_generation folder and performs mesh
subdivision using PyMesh.
Automatically aligns the eigenvectors using the KL divergence of the histograms.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file

Note: requires the --dataset_dir flag to be specified as well.
"""

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pymesh
import pytorch_lightning as pl
import torch
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import r2_score

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.plotting.figures import draw_mesh, draw_pc
from src.utils.data_generation import get_fourier, load_mesh, mesh_to_graph
from src.utils.eigenvectors import align_eigenvectors_kl
from src.utils.get_predictions import get_batched_predictions

# Read arguments
parser = ArgumentParser()
parser.add_argument("checkpoint", type=str)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphINR.add_model_specific_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
args = parser.parse_args()

# Data
dataset = GraphDataset(**vars(args))
mesh_train = load_mesh("data_generation/bunny/reconstruction/bun_zipper.ply")
u_train = dataset.get_inputs(0).numpy()
y_train = dataset.get_target(0).numpy()

# Plot training signal
rot = R.from_euler("xyz", [90, 00, 145], degrees=True).as_matrix()
fig = draw_mesh(mesh_train, intensity=y_train[:, 0], colorscale="Reds", rot=rot)
fig.update_layout(scene_camera=dict(eye=dict(x=1.1, y=1.1, z=0.2)))
fig.show()

# Model
model = GraphINR.load_from_checkpoint(args.checkpoint)

# Plot training predictions
inputs = torch.from_numpy(u_train).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_train, intensity=pred, colorscale="Reds", rot=rot)
fig.update_layout(scene_camera=dict(eye=dict(x=1.1, y=1.1, z=0.2)))
fig.show()

# Get test data and align eigenvectors to training ones
mesh_test = pymesh.subdivide(mesh_train, order=1)
_, adj_test = mesh_to_graph(mesh_test)
u_test = get_fourier(adj_test, k=args.n_fourier)
u_test = align_eigenvectors_kl(u_train, u_test)

# Predict signal
inputs = torch.from_numpy(u_test).float()
_, pred = get_batched_predictions(model, inputs, 0)

# Plot test signal
fig = draw_mesh(
    mesh_test,
    intensity=pred,
    colorscale="Reds",
    rot=rot,
    cmin=y_train.min(),
    cmax=y_train.max(),
)
fig.update_layout(scene_camera=dict(eye=dict(x=1.1, y=1.1, z=0.2)))
fig.show()

# Plot zoomed-in point clouds (take screenshots here!)
zoom = 0.6  # Lower is more zoomed
inputs = torch.from_numpy(u_train).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_train, rot=rot, color="black")
fig.update_layout(scene_camera=dict(eye=dict(x=zoom, y=zoom, z=0.2)))
pc_trace = draw_pc(
    mesh_train.vertices * 1.001,
    color=pred[:, 0],
    colorscale="Reds",
    rot=rot,
    marker_size=1.5,
).data[0]
fig.add_trace(pc_trace)
fig.write_html("super_resolution_bunny_original.html")
fig.show()

inputs = torch.from_numpy(u_test).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_test, rot=rot, color="black")
fig.update_layout(scene_camera=dict(eye=dict(x=zoom, y=zoom, z=0.2)))
pc_trace = draw_pc(
    mesh_test.vertices * 1.001,
    color=pred[:, 0],
    colorscale="Reds",
    rot=rot,
    marker_size=1.5,
).data[0]
fig.add_trace(pc_trace)
fig.write_html("super_resolution_bunny_superresolved.html")
fig.show()

# Compute squared error per node
mse = (y_train - pred[: u_train.shape[0]]) ** 2

# Plot error per node
fig = draw_mesh(mesh_train, intensity=mse, colorscale="Reds", rot=rot)
fig.update_layout(scene_camera=dict(eye=dict(x=1.1, y=1.1, z=0.2)))
fig.write_html("super_resolution_error.html")
fig.show()

# Compute r2
score = r2_score(y_train, pred[: u_train.shape[0]])
print(f"R2 score: {score}")

# Compute r2 without 90th percentile outliers
mask = mse < np.percentile(mse, 90)
r2_score_adjusted = r2_score(y_train[mask], pred[: u_train.shape[0]][mask])
print(f"R2 score adjusted (90th percentile): {r2_score_adjusted}")

# Compute r2 without 95th percentile outliers
mask = mse < np.percentile(mse, 95)
r2_score_adjusted = r2_score(y_train[mask], pred[: u_train.shape[0]][mask])
print(f"R2 score adjusted (95th percentile): {r2_score_adjusted}")

# Plot distribution of squared error
plt.figure(figsize=(2.2, 2.2))
plt.hist(mse, bins=10, density=True)
plt.yscale("log")
plt.xlabel("Squared error")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("super_resolution_error_density.pdf", bbox_inches="tight")
