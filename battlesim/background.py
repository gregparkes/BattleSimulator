#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:58:40 2019

@author: gparkes

This file handles 'background' drawing on to the lovely matplotlib canvas.
"""

import itertools as it
import numpy as np
from matplotlib import patches

from . import utils


def uniform_tile(xmin, xmax, ymin, ymax, tile_size=1, colors="random", alpha=.1):
    """
    Randomly tiles an area between xmin-xmax and ymin-ymax with random colours.
    By default this is a light background.

    Parameters
    -------
    xmin : int
        The x-minimum of the plot
    xmax : int
        The x-maximum of the plot
    ymin: int
        The y-minimum of the plot
    ymax : int
        The y-maximum of the plot
    tile_size : int
        The size of each tile.
    colors : str
        choose from [random, cycle], if cycle draws from utils.colorwheel
    alpha : float
        alpha of all the tiles

    Returns
    -------
    T : list of patches.Rectangle
        rectangles to add to a matplotlib.plot
    """
    valid_colors = ["random", "cycle"]

    if colors not in valid_colors:
        raise ValueError("colors option '{}' not found in {}".format(colors, valid_colors))

    T = []
    cyc = it.cycle(utils.colorwheel())

    for i in range(int(xmin)-1, int(xmax)+1, int(tile_size)):
        for j in range(int(ymin)-1, int(ymax)+1, int(tile_size)):
            if colors=="random":
                T.append(patches.Rectangle((i, j), tile_size, tile_size, fill=True, color=np.random.rand(3), alpha=alpha))
            elif colors=="cycle":
                T.append(patches.Rectangle((i, j), tile_size, tile_size, fill=True, color=next(cyc), alpha=alpha))
    return T



