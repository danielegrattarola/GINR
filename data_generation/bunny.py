import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from implicit_graphs.plotting import figures
from implicit_graphs.utils.data_generation import (get_fourier, get_output_dir,
                                                   load_mesh, mesh_to_graph,
                                                   normalized_laplacian)

# Load data
mesh = load_mesh("./data_generation/bunny/reconstruction/bun_zipper.ply")
points, adj = mesh_to_graph(mesh)
n = points.shape[0]

# Target signal
# Evolve reaction-diffusion model
Du, Dv, F, k = 0.16 * 4, 0.08 * 4, 0.060, 0.062
np.random.seed(1234)
lap = -normalized_laplacian(adj)  # Just for the diffusion
u = 0.2 * np.random.random(n) + 1
v = 0.2 * np.random.random(n)

n_iter = 30000
for i in tqdm(range(n_iter)):
    uvv = u * v * v
    u += Du * lap.dot(u) - uvv + F * (1 - u)
    v += Dv * lap.dot(v) + uvv - (F + k) * v

# Plots
rot = R.from_euler("xyz", [90, 00, 145], degrees=True).as_matrix()
fig = figures.draw_mesh(mesh, v, rot=rot, colorscale="Reds")
fig.show()

# Get Fourier features
u = get_fourier(adj)

output_dir = get_output_dir("bunny_v1/npz_files")
np.savez(
    os.path.join(output_dir, "data.npz"),
    points=points,
    fourier=u,
    target=v[:, None],
    faces=mesh.faces,
)
