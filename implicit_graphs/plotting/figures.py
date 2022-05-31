import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

PLOT_CONFIGS = {
    "bunny": {
        "colorscale": "Reds",
        "rot": R.from_euler("xyz", [90, 00, 145], degrees=True).as_matrix(),
        "lower_camera": True,
    },
    "protein_1AA7_A": {
        "colorscale": "RdBu",
        "rot": R.from_euler("xyz", [0, 180, 60], degrees=True).as_matrix(),
        "lower_camera": False,
    },
}


def draw_graph(points, adj, color=None):
    x, y, z = points.T
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker_size=3,
        marker_color=color,
        marker_colorbar=dict(thickness=20),
    )

    row = adj.tocoo().row
    col = adj.tocoo().col
    edge_endpoints = np.concatenate(
        [points[row], points[col], points[col] * np.nan],
        axis=-1,
    ).reshape(-1, 3)
    x, y, z = edge_endpoints.T
    edge_trace = go.Scatter3d(x=x, y=y, z=z, mode="lines", line_color="#aaaaaa")

    fig = go.Figure([node_trace, edge_trace])

    return fig


def draw_mesh(
    mesh, intensity=None, rot=None, colorscale=None, lower_camera=True, **kwargs
):
    points = mesh.vertices
    if rot is not None:
        points = np.array([rot @ p for p in points])
    x, y, z = points.T
    i, j, k = mesh.faces.T
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                intensity=intensity,
                i=i,
                j=j,
                k=k,
                colorscale=colorscale,
                **kwargs
            )
        ]
    )
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    if lower_camera:
        fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.2)))
    return fig


def draw_pc(
    points,
    color=None,
    rot=None,
    marker_size=3,
    colorscale=None,
    lower_camera=False,
    **kwargs
):
    if rot is not None:
        points = np.array([rot @ p for p in points])
    x, y, z = points.T
    fig = go.Figure(
        [
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(color=color, size=marker_size, colorscale=colorscale),
                **kwargs
            )
        ]
    )
    if lower_camera:
        fig.update_layout(scene_camera=dict(eye=dict(x=1.5, y=1.5, z=0.2)))
    return fig


def draw_pc_2d(points, color=None):
    x, y = points.T
    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker_size=3,
        marker_color=color,
        marker_colorbar=dict(thickness=20),
        marker_colorscale="Reds",
    )

    fig = go.Figure([node_trace])

    return fig
