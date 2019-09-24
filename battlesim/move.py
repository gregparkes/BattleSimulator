#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:16 2019

@author: gparkes

This file is concerned with movement functions associated with each army group.

There is the default 'move' and fancier options such as A*.
"""

from numba import jit

__all__ = ["default"]


@jit(nopython=True)
def default(M_pos, M_speed, dir_vec, distance, i, terrain_mod=1.):
    """
    Moves according to the euclidean distance from the target.

    We include a 'terrain' modifier to indicate global movement speed modification
    depending on the ground, for example.
    """
    # updates M_pos inplace
    M_pos[i] += (M_speed[i] * (dir_vec[i] / distance[i]) * terrain_mod)
    return
