#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:45:50 2019

@author: gparkes

This file determines different background 'terrains' to have for any given map.
This can be fixed size or infinite if it follows some mathematical function.
"""
import numpy as np
from scipy import stats

from . import utils


def _generate_random_gauss(pos, dim, res=1.):
    # fetch dimensions
    xmin, xmax, ymin, ymax = dim
    Nx = (xmax - xmin); Ny = (ymax - ymin)
    #scaling factors
    sx = Nx/4; sy = Ny/4
    # define mean
    m = np.random.rand(2) * np.array([Nx, Ny]) - np.array([xmax, ymax])
    # Diagonal elements for covariance matrix
    D = np.random.rand(2) + np.array([1., 0.])
    # random choice to invert cov
    R = np.random.choice([1., -1.])
    # construct covariance
    C = np.array([[D[0]*sx, D[1]*R], [D[1]*R, D[0]*sy]])
    # use scipy.stats
    z = stats.multivariate_normal(m, C).pdf(pos)
    return z


def _generate_random_terrain(xmin, xmax, ymin, ymax, resolution=1., n_gauss=100):

    Nx = (xmax - xmin); Ny = (ymax - ymin)
    # define meshgrid
    x, y, = np.mgrid[xmin:xmax:resolution, ymin:ymax:resolution]

    # create empty positions
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y

    contours = [_generate_random_gauss(pos, (xmin, xmax, ymin, ymax), resolution) for i in range(n_gauss)]

    Z_s = np.empty(Nx, Ny, n_gauss)
    for i, cont in enumerate(contours):
        Z_s[:, :, i] = cont
    # add probabilities
    Z = Z_s.sum(axis=2)
    # scale between 0 and 1 and return
    return utils.minmax(Z)


class Terrain(object):
    """
    The Terrain object is responsible for generating a terrain or map
    for the background of a given simulation.

    The Z-variable (depth) can be used in movement calculations of units, depending
    on which square they are walking on.

    In addition, the Z-variable could be used in calculations of the range/damage
    etc.

    The terrain object is responsible not only for the type of background but the
    way in which it is displayed.
    """
    def __init__(self, dim, form="random"):
        """
        Defines a Terrain object.

        Parameters
        -------
        dim : tuple (4,)
            x-min, x-max, y-min and y-max dimensions of the associated area.
        form : str
            Determines what type of terrain to generate. Choose from ['random', 'grid', 'contour']
                -random will choose 'random' points in the map to elevate into circular hills (essentially)
                -grid applies a function z = f(x, y) and discretizes this into square boxes.
                -contour applies a function z = f(x, y) and plots filled-contours of the background.

        """


