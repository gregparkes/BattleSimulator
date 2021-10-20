""" Various utilities for fast computation of things within the simulation. """

import numpy as np
from numba import njit, prange


@njit
def boundary_check(bxmin, bxmax, bymin, bymax, X, Y):
    """performs boundary checks on our locations movement inplace"""
    X[X < bxmin] = bxmin
    X[X > bxmax] = bxmax
    Y[Y < bymin] = bymin
    Y[Y > bymax] = bymax


@njit
def euclidean_distance(dX, dY):
    """ Given dX, dY vectors, compute distances D vector from M[x] and M[y]"""
    D = np.empty_like(dX)
    for i in prange(D.shape[0]):
        D[i] = np.sqrt(dX[i]*dX[i] + dY[i]*dY[i])
    return D


@njit
def sq_euclidean_distance(dX, dY):
    """ squared euclidean distance, cheaper. """
    D = np.empty_like(dX)
    for i in prange(D.shape[0]):
        D[i] = dX[i] * dX[i] + dY[i] * dY[i]
    return D


@njit
def sq_euclidean_distance2(X, Y, i, e_indices):
    """squared euclidean distance between X[i] and Y[i] and X[indices]/Y[indices]"""
    return sq_euclidean_distance(X[i] - X[e_indices], Y[i] - Y[e_indices])


@njit
def sq_distance_matrix(X, Y):
    """Where X, Y are vectors of (n,) length. """
    D = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
    # inplace op
    for i in prange(D.shape[0]):
        for j in range(i + 1, Y.shape[0]):
            dX = X[i] - X[j]
            dY = Y[i] - Y[j]
            sq_dist = dX*dX + dY*dY
            # set transpose also.
            D[i, j] = D[j, i] = sq_dist
    return D


@njit
def matrix_argmin(X):
    """ Calculates argmin along axis=1 dim. """
    A = np.empty(X.shape[0], dtype=np.int64)
    for i in prange(X.shape[0]):
        A[i] = np.argmin(X[i, :])
    return A


@njit
def minmax(X):
    """ Scales X into the [0, 1] range. """
    xm = np.min(X)
    return (X - xm) / (np.max(X) - xm)


@njit
def no_mean(X):
    """ Remove the mean from every value in X. """
    return X - np.mean(X)