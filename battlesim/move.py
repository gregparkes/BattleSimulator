#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:16 2019

@author: gparkes

This file is concerned with movement functions associated with each army group.

There is the default 'move' and fancier options such as A*.
"""

from numba import jit

__all__ = ["to_enemy"]

"""
Parameters
    pos: (np.ndarray, (n, 2), float all) positions of every unit
    speed: (np.ndarray, (n,) float all) base speed of every unit
    dd: (np.ndarray, (n,) float all) directional derivative to target
    distances: (np.ndarray, (n,) float all) unit distance to target
    i : (int) the current unit's index
"""

@jit(nopython=True)
def to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i):
    """
    Moves according to the euclidean distance towards the target.

    Returns adjustment, does not modify pos inplace.

    The update speed is calculated as:
        <unit base speed> * (<unit directional derivative> / <magnitude/distance>) * <terrain modifier>
    """
    return (speed[i] * (dd[i] / distances[i])) * (1. - (Z[Z_xi[i], Z_yi[i]] / 2.))
