import plotly.graph_objects as go


def scatter_3d(points, color=None):
    x, y, z = points.T
    return go.Scatter3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        mode="markers",
        marker=dict(size=1, cmid=0, color=color),
    )
