#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:45:50 2019

@author: gparkes

This file determines different background 'terrains' to have for any given map.
This can be fixed size or infinite if it follows some mathematical function.
"""
import numpy as np
from scipy.stats import multivariate_normal
from . import jitcode
from . import utils

def get_tile_size(dim, res):
    return int(np.abs(dim[0]-dim[1])//res), int(np.abs(dim[2]-dim[3])//res)


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
    sx = Nx/lm(Nx,slope,intercept); sy = Ny/lm(Ny,slope,intercept)
    # define mean
    m = np.random.rand(2) * np.array([Nx, Ny]) + np.array([xmin, ymin])
    # Diagonal elements for covariance matrix
    D = np.random.rand(2) + np.array([1., 0.])
    # random choice to invert cov
    R = np.random.choice([1., -1.])
    # construct covariance
    C = np.array([[D[0]*sx, D[1]*R], [D[1]*R, D[0]*sy]])
    # use scipy.stats
    z = multivariate_normal(m, C).pdf(pos)
    return z


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
    def __init__(self, dim=(0, 10, 0, 10), res=.1, form="contour"):
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
                -grid applies a function z = f(x, y) or random and discretizes this into square boxes.
                -contour applies a function z = f(x, y) or random and plots filled-contours of the background.
                -None defines a flat grid with no height penalties.
        """
        self.form_ = form
        self.bounds_ = dim
        self.res_ = res
        self.Z_ = None

    """ '################################## ATTRIBUTES ########################################### """


    def _get_form(self):
        return self._form
    def _set_form(self, f):
        if f not in [None, "grid", "contour"]:
            raise AttributeError("'form' must be [None, grid, contour]")
        self._form = f

    def _get_bounds(self):
        return self._bounds
    def _set_bounds(self, dim):
        if not isinstance(dim, (list, tuple)):
            raise TypeError("'dim' must be of type [list, tuple]")
        if len(dim) != 4:
            raise AttributeError("'dim' must contain 4 elements")
        if dim[1] <= dim[0]:
            raise AttributeError("xmax cannot be <= xmin")
        if dim[3] <= dim[2]:
            raise AttributeError("ymax cannot be <= ymin")
        utils.is_ntuple(dim, (int, float), (int, float), (int, float), (int, float))
        self._bounds = dim

    def _get_res(self):
        return self._res
    def _set_res(self, r):
        if not isinstance(r, (float, np.float)):
            raise TypeError("'res' must be of type [float]")
        if r < 1e-8:
            raise ValueError("'res' cannot be less than 0")
        self._res = r


    form_ = property(_get_form, _set_form, "The type of terrain")
    bounds_ = property(_get_bounds, _set_bounds, "Boundaries of the terrain")
    res_ = property(_get_res, _set_res, "resolution of the terrain")

    ############################## HIDDEN FUNCTIONS ################################################

    def _generate_random_terrain(self, n_gauss=100):
        xmin, xmax, ymin, ymax = self.bounds_
        # define meshgrid
        X, Y = self.get_grid()

        # create empty positions
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y

        Z_cont = np.stack(([_generate_random_gauss(pos, self.bounds_, self.res_) for i in range(n_gauss)]), axis=2)

        # add probabilities
        Z = Z_cont.sum(axis=2)
        # scale between 0 and 1 and return
        return jitcode.minmax(Z)


    def __repr__(self):
        return "Terrain(init={}, type='{}', dim={}, resolution={:0.3f})".format(
                self.Z_ is not None, self.form_, self.bounds_, self.res_)


    ##################################### FUNCTIONS ##################################################


    def get_grid(self):
        return np.mgrid[self.bounds_[0]:self.bounds_[1]:self.res_,
                        self.bounds_[2]:self.bounds_[3]:self.res_]


    def get_flat_grid(self):
        X, Y = self.get_grid()
        # they are repeats, return single.
        return X[:, 0], Y[0, :]


    def m_size(self):
        return (int((self.bounds_[1] - self.bounds_[0]) / self.res_),
                int((self.bounds_[3] - self.bounds_[2]) / self.res_))


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
        if self.form_ is None:
            self.Z_ = np.zeros(self.m_size())
        if f is None:
            self.Z_ = self._generate_random_terrain(n_random)
        elif callable(f):
            self.Z_ = jitcode.minmax(f(*self.get_grid()))
        else:
            raise TypeError("'f' must be a function or None")
        return self


    def plot(self, ax, **kwargs):
        # given an axes, plot the terrain using the parameters given.
        if self.Z_ is None:
            raise ValueError("Terrain not instantiated, call generate()")

        if self.form_ == "grid":
            xmin, xmax, ymin, ymax = self.bounds_
            ax.imshow(self.Z_,
                            aspect="auto", cmap="binary", **kwargs)
        elif self.form_ == "contour" or self.form_ == "random":
            X, Y = self.get_grid()
            ax.contourf(X, Y, self.Z_, cmap="binary", **kwargs)
        return
