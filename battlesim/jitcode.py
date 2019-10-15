#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:06:17 2019

@author: gparkes
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def distance_matrix(X):
    """ Assumes X is 2-dimensional """
    D = np.zeros((X.shape[0], X.shape[0]), dtype=np.float64)
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            D[i, j] = np.sqrt((X[i, 0] - X[j, 0])**2 + (X[i, 1] - X[j, 1])**2)
    # flip D and add those values to it.
    D += D.transpose()
    return D


@jit(nopython=True)
def matrix_argmin(X):
    """ Calculates argmin along the axis=1 dimension """
    A = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(X.shape[0]):
        A[i] = np.argmin(X[i, :])
    return A


@jit(nopython=True)
def boundary_check(bxmin, bxmax, bymin, bymax, pos):
    """performs boundary checks on our Mpos movement inplace"""
    pos[pos[:, 0] <= bxmin, 0] = bxmin
    pos[pos[:, 0] >= bxmax, 0] = bxmax
    pos[pos[:, 1] <= bymin, 1] = bymin
    pos[pos[:, 1] >= bymax, 1] = bymax


@jit(nopython=True)
def euclidean_distance(M):
    D = np.zeros(M.shape[0])
    for i in range(M.shape[0]):
        D[i] = np.sqrt((M[i, 0]*M[i, 0]) + (M[i, 1]*M[i, 1]))
    return D


@jit(nopython=True)
def direction_norm(M):
    distance = euclidean_distance(M)
    return (M.T/distance).T


@jit(nopython=True)
def create_grid(xmin,xmax,ymin,ymax,res):
    return np.mgrid[xmin:xmax:res, ymin:ymax:res]


@jit(nopython=True)
def minmax(X):
    X_min = np.min(X)
    return (X -X_min) / (np.max(X) -X_min)


@jit(nopython=True)
def remove_bias(x):
    return x - np.mean(x)