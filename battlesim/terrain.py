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
import matplotlib.pyplot as plt

from . import utils


def get_tile_size(dim, res):
    return int(np.abs(dim[0]-dim[1])//res), int(np.abs(dim[2]-dim[3])//res)


def get_grid(dim, res):
    return np.mgrid[dim[0]:dim[1]:res, dim[2]:dim[3]:res]


def _generate_random_gauss(pos, dim, res=1.):
    lm = lambda x, m, b: x*m+b
    # fetch dimensions
    xmin, xmax, ymin, ymax = dim
    Nx = (xmax - xmin); Ny = (ymax - ymin)
    #scaling factors
    # perfect scaling factor based on linear model y=mx + b
    x_n = np.arange(0, 300, 5); y_n = np.linspace(5, 2, 60)
    slope, intercept = np.polyfit(x_n, y_n, deg=1)
    # calculate scaling factor from formula
    sx = Nx/lm(Ny,slope,intercept); sy = Ny/lm(Ny,slope,intercept)
    # define mean
    m = np.random.rand(2) * np.array([Nx, Ny]) + np.array([xmin, ymin])
    # Diagonal elements for covariance matrix
    D = np.random.rand(2) + np.array([1., 0.])
    # random choice to invert cov
    R = np.random.choice([1., -1.])
    # construct covariance
    C = np.array([[D[0]*sx, D[1]*R], [D[1]*R, D[0]*sy]])
    # use scipy.stats
    z = stats.multivariate_normal(m, C).pdf(pos)
    return z


def _generate_random_terrain(dim, resolution=1., n_gauss=100):

    xmin, xmax, ymin, ymax = dim
    # define meshgrid
    X, Y = get_grid(dim, resolution)

    # create empty positions
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y

    Z_cont = np.stack(([_generate_random_gauss(pos, dim, resolution) for i in range(n_gauss)]), axis=2)

    # add probabilities
    Z = Z_cont.sum(axis=2)
    # scale between 0 and 1 and return
    return utils.minmax(Z)


def _map_function(dim, res, f):
    return utils.minmax(f(*get_grid(dim,res)))


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
    def __init__(self, dim, res=.1, form="contour"):
        """
        Defines a Terrain object.

        Parameters
        -------
        dim : tuple (4,)
            x-min, x-max, y-min and y-max dimensions of the associated area.
        res : float
            The resolution of the map. Higher resolutions are slower but more accurate.
        form : str
            Determines what type of terrain to generate. Choose from ['grid', 'contour']
                -random will choose 'random' points in the map to elevate into circular hills (essentially)
                -grid applies a function z = f(x, y) and discretizes this into square boxes.
                -contour applies a function z = f(x, y) and plots filled-contours of the background.
        """
        self.form_ = form
        self.dim_ = dim
        self.res_ = res
        self.Z_ = None


    def generate(self, f=None, n_random=50):
        """
        Generates the terrain using a function.

        Parameters
        -------
        f : function or None
            If None, uses 'random', else uses a map_function if ['grid', 'contour']

        Returns
        -------
        Z : np.ndarray
            The contour grid for the plane.
        """
        if f is None:
            self.Z_ = _generate_random_terrain(self.dim_, self.res_, n_random)
        else:
            self.Z_ = _map_function(self.dim_, self.res_, f)

        return self


    def plot(self, ax, **kwargs):
        # given an axes, plot the terrain using the parameters given.
        if self.Z_ is None:
            raise ValueError("Terrain not instantiated, call generate()")

        if self.form_ == "grid":
            xmin, xmax, ymin, ymax = self.dim_
            cax = ax.imshow(self.Z_,
                            aspect="auto", cmap="binary", **kwargs)
        elif self.form_ == "contour" or self.form_ == "random":
            X, Y = get_grid(self.dim_, self.res_)
            cax = ax.contourf(X, Y, self.Z_, cmap="binary", **kwargs)
        return cax


    def __repr__(self):
        return "Terrain(init={}, type='{}', dim={}, resolution={:0.3f})".format(
                self.Z_ is not None, self.form_, self.dim_, self.res_)
