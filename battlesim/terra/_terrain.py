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
from typing import Optional, Tuple

from battlesim._mathutils import minmax
from ._noise import create_perlin_map
from battlesim import _utils


def get_tile_size(dim: Tuple[float, float, float, float],
                  res: float):
    """Returns the tile size of the resolution."""
    return int(np.abs(dim[0] - dim[1]) // res), int(np.abs(dim[2] - dim[3]) // res)


class Terrain:
    """
    The Terrain object is responsible for generating a terra or map
    for the background of a given simulation.

    The Z-variable (depth) can be used in movement calculations of units, depending
    on which square they are walking on.

    In addition, the Z-variable could be used in calculations of the range/damage
    etc.

    The terra object is responsible not only for the type of background but the
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
            Determines what type of terra to generate. Choose from ['grid', 'contour']
                -grid applies a function z = f(x, y) or random and discretizes this into square boxes.
                -contour applies a function z = f(x, y) or random and plots filled-contours of the background.
                -None defines a flat grid with no height penalties.
        dtype : str
            Defines which function to use to generate noise map. Only `perlin` is currently implemented.
        """
        self.form_ = form
        self.bounds_ = dim
        self.res_ = res
        self._Z = None
        self.dtype = dtype

    """ '################################## ATTRIBUTES ########################################### """

    @property
    def form_(self):
        """Determines the appearance of the terra. Either grid or contour."""
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
        _utils.is_ntuple(dim, *([(int, float, np.float32, np.int32)] * 4))
        self._bounds = dim

    @property
    def res_(self):
        """The resolution of the Terrain."""
        return self._res

    @res_.setter
    def res_(self, r: float):
        if not isinstance(r, float):
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
        x0, x1, y0, y1 = self.bounds_
        return (int((x1-x0) / self.res_),
                int((y1-y0) / self.res_))

    def __repr__(self):
        return "Terrain(init={}, dtype='{}', dims={}, resolution={:0.3f})".format(
            self.Z_ is not None, self.dtype, self.bounds_, self.res_)

    """##################################### FUNCTIONS ##################################################"""

    def get_grid(self):
        x0, x1, y0, y1 = self.bounds_
        """Returns the grid as an mgrid."""
        x, y = np.linspace(x0, x1, self.Z_.shape[0]), np.linspace(y0, y1, self.Z_.shape[1])
        return np.meshgrid(x, y)

    def generate(self, f=None):
        """
        Generates the terra using a function.

        Parameters
        ----------
        f : function or None
            If None, uses 'random', else uses a map_function if ['grid', 'contour']

        Returns
        -------
        Z : np.ndarray
            The contour grid for the plane.
        """
        if self.form_ is None:
            self._Z = np.zeros(self._m_size())
        if f is None:
            dx, dy = self._m_size()
            self._Z = minmax(create_perlin_map(dx, dy, scale=dx // 3))
        elif callable(f):
            self._Z = minmax(f(*self.get_grid()))
        else:
            raise TypeError(f"'f' must be a function or None, not {f}")
        return self

    def plot(self, ax=None, **kwargs):
        """Plots the terra as a contour to visualize."""
        # given an axes, plot the terra using the parameters given.
        if self.Z_ is None:
            raise ValueError("Terrain not instantiated, call generate()")
        if ax is None:
            fig, ax = subplots(figsize=(8, 6))

        if self.form_ == "grid":
            ax.imshow(self.Z_, aspect="auto", cmap="binary", extent=self.bounds_, **kwargs)
        elif self.form_ == "contour" or self.form_ == "random":
            X, Y = self.get_grid()
            ax.contourf(X.T, Y.T, self.Z_, cmap="binary", extent=self.bounds_, **kwargs)
