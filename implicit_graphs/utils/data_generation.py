import os
import warnings

import numpy as np
import pymesh
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence


def get_fourier(adj, k=100):
    l = laplacian(adj)
    _, u = sp.linalg.eigsh(l, k=k, which="SM")
    n = l.shape[0]
    u *= np.sqrt(n)

    return u


def load_mesh(path, remove_isolated_vertices=True):
    mesh = pymesh.load_mesh(path)
    if remove_isolated_vertices:
        mesh, _ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def edges_to_adj(edges, n):
    a = sp.csr_matrix(
        (np.ones(edges.shape[:1]), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    a = a + a.T
    a.data[:] = 1.0

    return a


def mesh_to_graph(mesh):
    points, edges = pymesh.mesh_to_graph(mesh)
    n = points.shape[0]
    adj = edges_to_adj(edges, n)

    return points, adj


def get_output_dir(name):
    datset_dir = os.path.abspath("./dataset/")
    output_dir = os.path.join(datset_dir, name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def degree_matrix(A):
    degrees = np.array(A.sum(1)).flatten()
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def degree_power(A, k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency(A, symmetric=True):
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)


def laplacian(A):
    return degree_matrix(A) - A


def normalized_laplacian(A, symmetric=True):
    if sp.issparse(A):
        I = sp.eye(A.shape[-1], dtype=A.dtype)
    else:
        I = np.eye(A.shape[-1], dtype=A.dtype)
    normalized_adj = normalized_adjacency(A, symmetric=symmetric)
    return I - normalized_adj


def sphere_to_cartesian(lat, lon):
    """
    Converts latitude and longitude coordinates in degrees, to x, y, z cartesian
    coordinates.
    """
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def cartesian_to_sphere(x, y, z, return_netcdf_0_360=True):
    """
    Converts x, y, z cartesian coordinates to latitude and longitude coordinates
    in degrees.
    N.B.: NetCDF uses a 0:360 range for longitude.
    If return_netcdf_0_360=True, then the longitude will be returned in the
    NetCDF range. Otherwise, the standard -180:180 range is used.
    """
    lat = np.arcsin(z)
    lon = np.arctan2(y, x)

    lat, lon = np.rad2deg(lat), np.rad2deg(lon)

    if return_netcdf_0_360:
        lon = lon % 360.0

    return lat, lon
