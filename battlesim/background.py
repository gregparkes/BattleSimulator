#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:58:40 2019

@author: gparkes

This file handles 'background' drawing on to the lovely matplotlib canvas.
"""

import numpy as np

__all__ = ["random_tile", "meshgrid_tile", "contour_tile"]


def draw_tiles(ax, X, dim, alpha=.1, **kwargs):
    xmin, xmax, ymin, ymax = dim
    ax.imshow(X, alpha=alpha, extent=[xmin, ymax, ymin, ymax], aspect="auto", **kwargs)
    return


def draw_contour(ax, x, y, z, fill=True, **kwargs):
    if fill:
        ax.contourf(x, y, z, **kwargs)
    else:
        ax.contour(x, y, z, **kwargs)
    return


def random_tile(ax,
                dim=(0, 30, 0, 30),
                alpha=.1):
    """
    Randomly tiles an area between xmin-xmax and ymin-ymax with random colours.
    By default this is a light background.

    Parameters
    -------
    ax : matplotlib.ax
        The axes
    dim : tuple (4,)
        xmin, xmax, ymin, ymax
    alpha : float
        alpha of all the tiles

    Returns
    -------
    None
    """
    xmin, xmax, ymin, ymax = dim
    X = np.random.rand(xmax-xmin, ymax-ymin, 3)
    # draw
    draw_tiles(ax, X, dim, alpha=alpha)


def meshgrid_tile(ax,
                  dim=(0, 30, 0, 30),
                  resolution=1.,
                  alpha=1.,
                  f = lambda x, y: x+y, cmap="binary_r"):
    """
    Tiles an area using a 'meshgrid function' between xmin-xmax and
    ymin-ymax using a function.

    Parameters
    -------
    ax : matplotlib.ax
        The axes
    dim : tuple (4,)
        xmin, xmax, ymin, ymax
    alpha : float
        alpha of all the tiles
    f : function
        A function taking two parameters x and y, to return a meshgrid over z dimension.

    Returns
    -------
    None
    """
    xmin, xmax, ymin, ymax = dim
    x, y = np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # draw
    draw_tiles(ax, Z, dim, alpha=alpha, cmap=cmap)


def contour_tile(ax,
                 dim=(0, 30, 0, 30),
                 alpha=1.,
                 resolution=.5,
                 fill=True,
                 f = lambda x, y: x+y, cmap="binary_r"):
    """
    Tiles an area using a 'contour function' between xmin-xmax and
    ymin-ymax using a function.

    Parameters
    -------
    ax : matplotlib.ax
        The axes
    dim : tuple (4,)
        xmin, xmax, ymin, ymax
    alpha : float
        alpha of all the tiles
    fill : bool
        If true, uses contourf, else uses contour
    f : function
        A function taking two parameters x and y, to return a meshgrid over z dimension.

    Returns
    -------
    None
    """
    xmin, xmax, ymin, ymax = dim
    x, y = np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # draw
    draw_contour(ax, x, y, Z, fill, cmap=cmap, alpha=alpha)
