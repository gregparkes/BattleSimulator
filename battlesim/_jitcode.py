#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:06:17 2019

@author: gparkes
"""

import numpy as np
from numba import njit


@njit
def _distance_matrix_calculation(X, D):
    """Determines the distance matrix (euclidean)."""
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            D[i, j] = np.sqrt((X[i, 0] - X[j, 0])**2 + (X[i, 1] - X[j, 1])**2)


@njit
def _distance_vector_calculation(X, D):
    """Determines the distance as a vector."""
    for i in range(X.shape[0]):
        D[i] = np.sqrt((X[i, 0]*X[i, 0]) + (X[i, 1]*X[i, 1]))


@njit
def _matrix_min(X, A):
    """Matrix minimization function."""
    for i in range(X.shape[0]):
        A[i] = np.argmin(X[i, :])


@njit
def distance_matrix(X):
    """ Assumes X is 2-dimensional """
    D = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    # inplace op
    _distance_matrix_calculation(X, D)
    # flip D and add those values to it.
    D += D.transpose()
    return D


@njit
def matrix_argmin(X):
    """Calculates argmin along the axis=1 dimension """
    A = np.zeros(X.shape[0], dtype=np.int64)
    _matrix_min(X, A)
    return A


@njit
def boundary_check(bxmin, bxmax, bymin, bymax, pos):
    """performs boundary checks on our Mpos movement inplace"""
    pos[pos[:, 0] < bxmin, 0] = bxmin
    pos[pos[:, 0] > bxmax, 0] = bxmax
    pos[pos[:, 1] < bymin, 1] = bymin
    pos[pos[:, 1] > bymax, 1] = bymax


@njit
def euclidean_distance(M):
    """Determine the euclidean distance between all vectors in M."""
    D = np.zeros(M.shape[0])
    _distance_vector_calculation(M,D)
    return D


@njit
def direction_norm(dd):
    """Determine the directional norm between all vectors in dd."""
    distance = euclidean_distance(dd)
    return (dd.T/distance).T


@njit
def create_grid(xmin,xmax,ymin,ymax,res):
    """Creates a numpy mgrid given bounds."""
    return np.mgrid[xmin:xmax:res, ymin:ymax:res]


@njit
def minmax(X):
    """Calculate the minmax within vector X."""
    X_min = np.min(X)
    return (X - X_min) / (np.max(X) - X_min)


@njit
def remove_mean(x):
    """Removes the mean from vector x."""
    return x - np.mean(x)