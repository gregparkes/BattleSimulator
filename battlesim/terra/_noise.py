#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: gparkes

This file contains perlin noise functions for terra.
"""

import numpy as np
from numba import njit, prange


@njit
def smooth_noise(noise, x, y, noisewidth=100, noiseheight=100):
    """Determines some smooth noise at point [x, y]"""
    # get fractional part
    fractX = x - int(x)
    fractY = y - int(y)
    # wrap
    x1 = (int(x) + noisewidth) % noisewidth
    y1 = (int(y) + noiseheight) % noiseheight
    # neighbor values
    x2 = (x1 + noisewidth - 1) % noisewidth
    y2 = (y1 + noiseheight - 1) % noiseheight
    # smooth the noise with bilinear interpolation
    val = 0.0
    val += fractX * fractY * noise[y1, x1]
    val += (1 - fractX) * fractY * noise[y1, x2]
    val += fractX * (1 - fractY) * noise[y2, x1]
    val += (1 - fractX) * (1 - fractY) * noise[y2, x2]
    return val


@njit
def turbulence(noise, x, y, size, dim_x, dim_y):
    """Adds turbulence at point [x, y]"""
    val = 0.0
    init_size = size
    while size >= 1:
        val += smooth_noise(noise, x / size, y / size, dim_x, dim_y) * size
        size /= 2.0
    return 128 * val / init_size


@njit
def create_perlin_map(dim_x, dim_y, scale=30):
    """Defines a perlin map with turbulence (upgrade on gauss map)

    It's recommended that scale is an integer close to 30% of one of the map dimensions.
    For example dims (100, 100) look nice with scale in the range [20, 50]

    Too high and there is no difference, too low and there is pixellized localities.

    """
    base_noise = np.random.rand(dim_x, dim_y)
    new_noise = np.random.rand(dim_x, dim_y)
    # set every value in alterable noise map
    for x in prange(dim_x):
        for y in prange(dim_y):
            new_noise[x, y] = turbulence(base_noise, x, y, scale, dim_x, dim_y)
    return new_noise
