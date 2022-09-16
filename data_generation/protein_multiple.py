import os
import shutil
from glob import glob

import numpy as np
from pytorch_lightning import seed_everything

from src.utils.data_generation import (get_fourier, get_output_dir, load_mesh,
                                       mesh_to_graph)

seed_everything(1234)

dataset_name = "proteins"
dataset_dir = f"./data_generation/{dataset_name}"

names = glob(f"{dataset_dir}/obj_files/*.obj")
names = [n.split("/")[-1][:-4] for n in names]

for name in names:
    print(f"Doing {name}")
    # Load data
    mesh_file = f"{dataset_dir}/obj_files/{name}.obj"
    mesh = load_mesh(mesh_file)
    points, adj = mesh_to_graph(mesh)

    # Target signal
    target = np.load(f"{dataset_dir}/npz_files/{name}.npz")["charges"]
    target = (target - target.mean()) / target.std()

    # Get Fourier features
    u = get_fourier(adj)

    output_dir = get_output_dir(f"{dataset_name}/npz_files")
    np.savez(
        os.path.join(output_dir, f"{name}.npz"),
        points=points,
        fourier=u,
        target=target[:, None],
    )

    output_dir = get_output_dir(f"{dataset_name}/obj_files")
    shutil.copy(mesh_file, output_dir)
