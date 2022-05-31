import os
import shutil

import numpy as np
from pytorch_lightning import seed_everything

from implicit_graphs.plotting import figures
from implicit_graphs.plotting.figures import PLOT_CONFIGS
from implicit_graphs.utils.data_generation import (get_fourier, get_output_dir,
                                                   load_mesh, mesh_to_graph)

seed_everything(1234)
dataset_name = "protein_1AA7_A"

# Load data
mesh_file = "./data_generation/proteins/obj_files/1AA7_A.obj"
mesh = load_mesh(mesh_file)
points, adj = mesh_to_graph(mesh)

# Target signal
target = np.load("./data_generation/proteins/npz_files/1AA7_A.npz")["charges"]
target = (target - target.mean()) / target.std()

# Plots
rot = PLOT_CONFIGS[dataset_name]["rot"]
fig = figures.draw_mesh(
    mesh,
    intensity=target,
    rot=rot,
    colorscale=PLOT_CONFIGS[dataset_name]["colorscale"],
    lower_camera=PLOT_CONFIGS[dataset_name]["lower_camera"],
)
fig.show()

# Get Fourier features
u = get_fourier(adj)

output_dir = get_output_dir(f"{dataset_name}/npz_files")
np.savez(
    os.path.join(output_dir, "data.npz"),
    points=points,
    fourier=u,
    target=target[:, None],
)

output_dir = get_output_dir(f"{dataset_name}/obj_files")
shutil.copy(mesh_file, output_dir)
