#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:16 2019

@author: gparkes

This file is concerned with movement functions associated with each army group.

There is the default 'move' and fancier options such as A*.
"""

from numba import jit

__all__ = ["euclidean"]


@jit(nopython=True)
def euclidean(M_pos, M_speed, direction, magnitude, i, terrain_mod=1.):
    """
    Moves according to the euclidean distance from the target.

    We include a 'terrain' modifier to indicate global movement speed modification
    depending on the ground, for example.

    Modifies M_pos inplace; no return.

    The update speed is calculated as:
        <unit base speed> * (<unit directional derivative> / <magnitude/distance>) * <terrain modifier>

    Parameters
    -------
    M_pos : np.ndarray (n, 2)
        Position of the units
    M_speed : np.ndarray (n,)
        Speed of the units
    direction : np.ndarray (n, 2)
        Directional derivative of units
    magnitude : np.ndarray (n,)
        Distance from target
    i : int
        Current unit index
    terrain_mod : float
        Global speed modifier to each unit

    Returns
    -------
    inplace
    """
    # updates M_pos inplace
    M_pos[i] += (M_speed[i] * (direction[i] / magnitude[i]) * terrain_mod)
    return
