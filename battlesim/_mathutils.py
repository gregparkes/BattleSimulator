""" Various utilities for fast computation of things within the simulation. """

import numpy as np
from numpy.typing import NDArray
from numba import jit


@jit
def boundary_check(
    bxmin: float,
    bxmax: float,
    bymin: float,
    bymax: float,
    x_pos: NDArray[np.float_],
    y_pos: NDArray[np.float_],
) -> None:
    """performs boundary checks on our locations movement inplace"""
    x_pos[x_pos < bxmin] = bxmin
    x_pos[x_pos > bxmax] = bxmax
    y_pos[y_pos < bymin] = bymin
    y_pos[y_pos > bymax] = bymax


@jit
def boundary_check2(bounds, X, Y):
    """performs boundary checks on our locations movement inplace"""
    X[X < bounds[0]] = bounds[0]
    X[X > bounds[1]] = bounds[1]
    Y[Y < bounds[2]] = bounds[2]
    Y[Y > bounds[3]] = bounds[3]


@jit
def euclidean_distance(
    delta_x: NDArray[np.float_], delta_y: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Given dX, dY vectors, compute distances D vector from M[x] and M[y]"""
    distance = np.empty_like(delta_x, dtype=np.float64)
    size = distance.shape[0]
    for i in range(size):
        distance[i] = np.sqrt(delta_x[i] * delta_x[i] + delta_y[i] * delta_y[i])
    return distance


@jit
def sq_euclidean_distance(
    delta_x: NDArray[np.float_], delta_y: NDArray[np.float_]
) -> NDArray[np.float_]:
    """squared euclidean distance, cheaper."""
    distance = np.empty_like(delta_x)
    size = distance.shape[0]
    for i in range(size):
        distance[i] = delta_x[i] * delta_x[i] + delta_y[i] * delta_y[i]
    return distance


@jit
def sq_euclidean_distance2(
    x_pos: NDArray[np.float_],
    y_pos: NDArray[np.float_],
    i: int,
    e_indices: NDArray[np.uint],
) -> NDArray[np.float_]:
    """squared euclidean distance between X[i] and Y[i] and X[indices]/Y[indices]"""
    return sq_euclidean_distance(
        x_pos[i] - x_pos[e_indices], y_pos[i] - y_pos[e_indices]
    )


@jit
def sq_distance_matrix(
    x_pos: NDArray[np.float_], y_pos: NDArray[np.float_]
) -> NDArray[np.float_]:
    """Where X, Y are vectors of (n,) length."""
    distance = np.zeros((x_pos.shape[0], y_pos.shape[0]), dtype=np.float64)
    # inplace op
    for i in range(distance.shape[0]):
        for j in range(i + 1, y_pos.shape[0]):
            delta_x = x_pos[i] - x_pos[j]
            delta_y = y_pos[i] - y_pos[j]
            sq_dist = delta_x * delta_x + delta_y * delta_y
            # set transpose also.
            distance[i, j] = distance[j, i] = sq_dist
    return distance


@jit
def matrix_argmin(x_pos: NDArray[np.float_]) -> NDArray[np.int64]:
    """Calculates argmin along axis=1 dim."""
    mins = np.empty(x_pos.shape[0], dtype=np.int64)
    for i in range(x_pos.shape[0]):
        mins[i] = np.argmin(x_pos[i, :])
    return mins


@jit
def minmax(mat: NDArray[np.float_]) -> NDArray[np.float_]:
    """Scales X into the [0, 1] range."""
    xm = np.min(mat)
    return (mat - xm) / (np.max(mat) - xm)


@jit
def no_mean(mat: NDArray[np.float_]) -> NDArray[np.float_]:
    """Remove the mean from every value in X."""
    return mat - np.mean(mat)
