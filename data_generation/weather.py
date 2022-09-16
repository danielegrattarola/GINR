import os
from datetime import datetime, timedelta

import getgfs
import numpy as np
import plotly.graph_objects as go
import pymesh
from pytorch_lightning import seed_everything
from scipy.spatial import ConvexHull

from src.utils.data_generation import (get_fourier, get_output_dir,
                                       mesh_to_graph, sphere_to_cartesian)

seed_everything(1234)

# Fetch all data
# 'dpt2m'   : 2 m above ground dew point temperature [k]
# 'tcdcclm' : entire atmosphere total cloud cover [%]
# 'gustsfc' : surface wind speed (gust) [m/s]
print("Fetching data")
variables = ["dpt2m", "tcdcclm", "gustsfc"]
colorscale = ["hot", "Blues", "Spectral"]
date = (datetime.now() - timedelta(1)).strftime("%Y%m%d")  # We used: "20220509"
lat_range = "[-90:90]"
lon_range = "[0:360]"
resolution = "1p00"
f = getgfs.Forecast(resolution)
result = f.get(variables, f"{date} 00:00", lat_range, lon_range)

# Get mesh
lat, lon = (
    np.array(result.variables[variables[0]].coords["lat"].values),
    np.array(result.variables[variables[0]].coords["lon"].values),
)
lats, lons = np.meshgrid(lat, lon)
lats, lons = lats.T, lons.T  # (lat, lon), (lat, lon)
xyz = sphere_to_cartesian(lats, lons)  # (lat, lon, 3)

# Get graph
print("Triangulation")
hull = ConvexHull(xyz.reshape(-1, 3))
mesh = pymesh.form_mesh(hull.points, hull.simplices)
points, adj = mesh_to_graph(mesh)

# Get Fourier features
print(f"Computing embeddings, size=({adj.shape})")
u = get_fourier(adj)

for idx, v in enumerate(variables):
    data = result.variables[v].data.transpose(1, 2, 0)  # (lat, lon, 1)
    data = (data - data.mean()) / data.std()

    # Save signals
    print("Saving signals")
    output_dir_name = f"weather_{v}"
    output_dir = get_output_dir(f"{output_dir_name}/npz_files")
    np.savez(
        os.path.join(output_dir, "data.npz"),
        points=points,
        fourier=u,
        target=data.reshape(-1, 1),
        faces=mesh.faces,
    )

    # Plots
    fig = go.Figure(
        go.Scattergeo(
            lat=lats.reshape(-1),
            lon=lons.reshape(-1),
            mode="markers",
            marker_symbol="square",
            marker_line_width=0,
            marker_color=data.reshape(-1),
            marker_opacity=0.5,
            marker_colorscale=colorscale[idx],
        )
    )
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
    img_path = os.path.join(get_output_dir(f"{output_dir_name}/imgs"), f"{v}.png")
    fig.write_image(img_path, width=1024 * 2, height=1024 * 2)
    os.system(f"convert -trim {img_path} {img_path}")

    # Plot mesh
    x, y, z = hull.points.T
    i, j, k = hull.simplices.T
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                intensity=data.reshape(-1),
                colorscale=colorscale[idx],
            )
        ]
    )
