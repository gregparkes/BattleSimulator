#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:45:50 2019

@author: gparkes

This file determines different background 'terrains' to have for any given map.
This can be fixed size or infinite if it follows some mathematical function.
"""
import numpy as np
from matplotlib.pyplot import subplots
from scipy.stats import multivariate_normal
from typing import Optional, Tuple

from numba import njit, prange


from ._jitcode import minmax
from ._noise import create_perlin_map
from . import utils


def get_tile_size(dim: Tuple[float, float, float, float],
                  res: float):
    """Returns the tile size of the resolution."""
    return int(np.abs(dim[0] - dim[1]) // res), int(np.abs(dim[2] - dim[3]) // res)


def _generate_random_gauss(pos: np.ndarray,
                           dim: Tuple[float, float, float, float]):
    def lm(x, sl, b):
        """Linear model function."""
        return x * sl + b

    # fetch dimensions
    xmin, xmax, ymin, ymax = dim
    Nx = (xmax - xmin)
    Ny = (ymax - ymin)
    # scaling factors
    # perfect scaling factor based on linear model y=mx + b
    x_n = np.arange(0, 300, 5)
    y_n = np.linspace(5, 3, 60)
    coef = np.polyfit(x_n, y_n, deg=1)
    # calculate scaling factor from formula
    sx = Nx / lm(Nx, coef[0], coef[1])
    sy = Ny / lm(Ny, coef[0], coef[1])

    # print(sx, sy)
    # define mean
    m = np.random.rand(2) * np.array([Nx, Ny]) + np.array([xmin, ymin])
    # Diagonal elements for covariance matrix
    D = np.random.rand(2) + np.array([4., 1.5])
    # random choice to invert cov
    R = np.random.choice([1., -1.])
    # construct covariance
    C = np.array([[D[0] * sx, D[1] * R], [D[1] * R, D[0] * sy]])

    #print(m, C)
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

    def __init__(self,
                 dim: Tuple[float, float, float, float] = (0., 10., 0., 10.),
                 res: float = .1,
                 form: Optional[str] = "contour",
                 dtype: Optional[str] = "perlin"):
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
        dtype : str
            Defines which function to use to generate noise map.
        """
        self.form_ = form
        self.bounds_ = dim
        self.res_ = res
        self._Z = None
        self.dtype = dtype

    """ '################################## ATTRIBUTES ########################################### """

    @property
    def form_(self):
        """Determines the appearance of the terrain. Either grid or contour."""
        return self._form

    @form_.setter
    def form_(self, f: Optional[str]):
        if f not in [None, "grid", "contour"]:
            raise AttributeError("'form' must be [None, grid, contour]")
        self._form = f

    @property
    def bounds_(self):
        """Determines the dimensions by which to calculate Terrain."""
        return self._bounds

    @bounds_.setter
    def bounds_(self, dim: Tuple[float, float, float, float]):
        if not isinstance(dim, (list, tuple)):
            raise TypeError("'dim' must be of type [list, tuple]")
        if len(dim) != 4:
            raise AttributeError("'dim' must contain 4 elements")
        if dim[1] <= dim[0]:
            raise AttributeError("xmax cannot be <= xmin")
        if dim[3] <= dim[2]:
            raise AttributeError("ymax cannot be <= ymin")
        utils.is_ntuple(dim, *([(int, float, np.float, np.int, np.float32, np.int32)] * 4))
        self._bounds = dim

    @property
    def res_(self):
        """The resolution of the Terrain."""
        return self._res

    @res_.setter
    def res_(self, r: float):
        if not isinstance(r, (float, np.float)):
            raise TypeError("'res' must be of type [float]")
        if r < 1e-8:
            raise ValueError("'res' cannot be less than 0")
        self._res = r

    @property
    def Z_(self):
        """The array defining height."""
        return self._Z

    """############################## HIDDEN FUNCTIONS ################################################"""

    def _m_size(self):
        return (int((self.bounds_[1] - self.bounds_[0]) / self.res_),
                int((self.bounds_[3] - self.bounds_[2]) / self.res_))

    def _generate_random_terrain(self, n_gauss: int = 3):
        # define meshgrid
        X, Y = self.get_grid()

        # create empty positions
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        Z_cont = np.stack(([_generate_random_gauss(pos, self.bounds_) for _ in range(n_gauss)]), axis=2)

        # add probabilities
        Z = Z_cont.sum(axis=2)
        # scale between 0 and 1 and return
        return minmax(Z)

    def __repr__(self):
        return "Terrain(init={}, type='{}', dim={}, resolution={:0.3f})".format(
            self.Z_ is not None, self.form_, self.bounds_, self.res_)

    """##################################### FUNCTIONS ##################################################"""

    def get_grid(self):
        """Returns the grid as an mgrid."""
        return np.mgrid[self.bounds_[0]:self.bounds_[1]:self.res_,
               self.bounds_[2]:self.bounds_[3]:self.res_]

    def get_flat_grid(self):
        """Produces a flat terrain grid."""
        X, Y = self.get_grid()
        # they are repeats, return single.
        return X[:, 0], Y[0, :]

    def generate(self, f=None, n_random: int = 50):
        """
        Generates the terrain using a function.

        Parameters
        ----------
        f : function or None
            If None, uses 'random', else uses a map_function if ['grid', 'contour']
        n_random : int, optional
            The number of random gaussians to generate.

        Returns
        -------
        Z : np.ndarray
            The contour grid for the plane.
        """
        if self.form_ is None:
            self._Z = np.zeros(self._m_size())
        if f is None:
            if self.dtype == 'gauss':
                self._Z = self._generate_random_terrain(n_random)
            elif self.dtype == "perlin":
                dx, dy = self._m_size()
                self._Z = minmax(create_perlin_map(dx, dy, scale=dx // 3))
            else:
                raise TypeError(f"'dtype' must be one of ['gauss', 'perlin'] not {self.dtype}")
        elif callable(f):
            self._Z = minmax(f(*self.get_grid()))
        else:
            raise TypeError(f"'f' must be a function or None, not {f}")
        return self

    def plot(self, ax=None, **kwargs):
        """Plots the terrain as a contour to visualize."""
        # given an axes, plot the terrain using the parameters given.
        if self.Z_ is None:
            raise ValueError("Terrain not instantiated, call generate()")
        if ax is None:
            fig, ax = subplots(figsize=(8, 6))

        if self.form_ == "grid":
            ax.imshow(self.Z_, aspect="auto", cmap="binary", extent=self.bounds_, **kwargs)
        elif self.form_ == "contour" or self.form_ == "random":
            X, Y = self.get_grid()
            ax.contourf(X, Y, self.Z_, cmap="binary", extent=self.bounds_, **kwargs)
