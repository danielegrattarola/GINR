import matplotlib.pyplot as plt
import numpy as np
import pymesh
from pytorch_lightning import seed_everything

from src.utils.data_generation import (get_fourier, get_output_dir,
                                       mesh_to_graph)
from src.utils.eigenvectors import align_eigenvectors_kl

seed_everything(1234)

variables = ["dpt2m", "tcdcclm", "gustsfc"]
n_fourier = 100

# Load original dataset from first variable
v = variables[0]
dataset_dir = f"./dataset/weather_time_{v}"
mesh = pymesh.load_mesh(f"{dataset_dir}/mesh.obj")
u = np.load(f"{dataset_dir}/fourier.npy")
p = np.load(f"{dataset_dir}/points.npy")

# Create super-resolved sphere
mesh_test = pymesh.subdivide(mesh, order=1, method="simple")
_, adj_test = mesh_to_graph(mesh_test)
print(f"Computing embeddings, size=({adj_test.shape})")
u_test = get_fourier(adj_test, k=n_fourier)
p_test = mesh_test.vertices

for i in range(n_fourier):
    plt.subplot(10, 10, i + 1)
    plt.hist(u[:, i], bins=100, alpha=0.5, density=True)
    plt.hist(u_test[:, i], bins=100, alpha=0.5, density=True)
plt.show()

# Align eigenvectors (use KL heuristic and sign heuristic)
print("Aligning eigenvectors using heuristics")
u_test_aligned = align_eigenvectors_kl(u, u_test)
switch = np.sign(u[0]) != np.sign(u_test[0])
u_test_aligned[:, switch] = -u_test_aligned[:, switch]
for i in range(n_fourier):
    if np.mean(np.sign(u_test_aligned[: u.shape[0], i]) == np.sign(u[:, i])) < 0.6:
        print(f"WARNING: Issue with alignment of feature {i}!!!")

# Cut eigenvectors
print("Cutting away useless eigenvectors")
u_test_aligned_cut = u_test_aligned[:, 65:99]

# Save files for all variables
for v in variables:
    dataset_dir_out = get_output_dir(f"weather_time_{v}_sr_cut")
    np.save(f"{dataset_dir_out}/points.npy", p_test)
    np.save(f"{dataset_dir_out}/fourier_sr.npy", u_test)
    np.save(f"{dataset_dir_out}/fourier_sr_aligned.npy", u_test_aligned)
    np.save(f"{dataset_dir_out}/fourier_sr_aligned_cut.npy", u)
