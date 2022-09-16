import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything

from src.utils.data_generation import get_fourier, get_output_dir

seed_everything(1234)

# Load data
edges = (
    np.loadtxt("./data_generation/US-county-fb/US-county-fb-graph.txt").astype(int) - 1
)
g = nx.from_edgelist(edges)
adj = nx.adjacency_matrix(g)

nodes = pd.read_csv(
    "./data_generation/US-county-fb/US-county-fb-2012-feats.csv"
).values[:, 1:]

# Target signal
target = nodes[:, -1]

# Get Fourier features
u = get_fourier(adj)

# Plots
nx.draw(
    nx.from_edgelist(edges),
    pos=u[:, 1:3],
    node_size=1,
    edge_color="#eeeeee",
    node_color=nodes[:, -1],
    cmap="RdBu",
    width=0.1,
)
plt.show()

output_dir = get_output_dir("us_elections/npz_files")
np.savez(
    os.path.join(output_dir, "data.npz"),
    points=nodes,
    fourier=u,
    target=target[:, None],
)
