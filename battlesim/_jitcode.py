#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:06:17 2019

@author: gparkes
"""

import numpy as np
from numba import jit, prange


@jit(nopython=True)
def sq_distance_matrix(X, Y):
    """ Assumes X is 2-dimensional """
    D = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float64)
    # inplace op
    for i in prange(D.shape[0]):
        for j in range(i + 1, Y.shape[0]):
            _dist = ((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            # set transpose also.
            D[i, j] = D[j, i] = _dist
    return D


@jit(nopython=True)
def matrix_argmin(X):
    """Calculates argmin along the axis=1 dimension """
    A = np.zeros(X.shape[0], dtype=np.int64)
    for i in prange(X.shape[0]):
        A[i] = np.argmin(X[i, :])
    return A


@jit(nopython=True)
def boundary_check(bxmin, bxmax, bymin, bymax, X, Y):
    """performs boundary checks on our Mpos movement inplace"""
    X[X < bxmin] = bxmin
    X[X > bxmax] = bxmax
    Y[Y < bymin] = bymin
    Y[Y > bymax] = bymax


@jit(nopython=True)
def euclidean_distance(M):
    """Determine the euclidean distance between all vectors in M."""
    D = np.zeros(M.shape[0])
    for i in prange(M.shape[0]):
        D[i] = np.sqrt((M[i, 0] * M[i, 0]) + (M[i, 1] * M[i, 1]))
    return D


def euclidean_distance2(dX, dY):
    """Splits M into dX and dY. Assumes dX and dY are same shape"""
    D = np.empty_like(dX)
    for i in prange(dX.shape[0]):
        D[i] = np.sqrt(dX[i]*dX[i] + dY[i]*dY[i])
    return D


@jit(nopython=True)
def sq_euclidean_distance2(X, Y, i, e_indices):
    """Calculates the square euclidean distance (only for distance comparison, not using sqrt is faster)"""
    D = np.zeros(e_indices.shape[0])
    # define distance to each enemy for unit i
    dX = X[i] - X[e_indices]
    dY = Y[i] - Y[e_indices]
    # make an array the shape of enemies
    for j in prange(D.shape[0]):
        # define distance from each unit to all others. (of shape e_indices)
        D[j] = dX[j]*dX[j] + dY[j]*dY[j]
    return D


@jit(nopython=True)
def direction_norm(dx, dy):
    """Determine the directional norm between all vectors in dd."""
    distance = euclidean_distance2(dx, dy)
    return (dd.T / distance).T


@jit(nopython=True)
def create_grid(xmin, xmax, ymin, ymax, res):
    """Creates a numpy mgrid given bounds."""
    return np.mgrid[xmin:xmax:res, ymin:ymax:res]


@jit(nopython=True)
def minmax(X):
    """Calculate the minmax within vector X."""
    X_min = np.min(X)
    return (X - X_min) / (np.max(X) - X_min)


@jit(nopython=True)
def remove_mean(x):
    """Removes the mean from vector x."""
    return x - np.mean(x)
