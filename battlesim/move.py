#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:16 2019

@author: gparkes

This file is concerned with movement functions associated with each army group.

There is the default 'move' and fancier options such as A*.
"""

from numba import jit

__all__ = ["to_enemy", "from_enemy"]


@jit(nopython=True)
def to_enemy(M_pos, M_speed, direction, magnitude, i, terrain_mod=1.):
    """
    Moves according to the euclidean distance towards the target.

    Modifies M_pos inplace; no return.

    The update speed is calculated as:
        <unit base speed> * (<unit directional derivative> / <magnitude/distance>) * <terrain modifier>
    """
    # updates M_pos inplace
    M_pos[i] += (M_speed[i] * (direction[i] / magnitude[i]) * terrain_mod)
    return


@jit(nopython=True)
def from_enemy(M_pos, M_speed, direction, magnitude, i, terrain_mod=1.):
    """
    Moves according to the euclidean distance from the target.

    Modifies M_pos inplace; no return.

    The update speed is calculated as:
        <unit base speed> * (<unit directional derivative> / <magnitude/distance>) * <terrain modifier>
    """
    # updates M_pos inplace
    M_pos[i] -= (M_speed[i] * (direction[i] / magnitude[i]) * terrain_mod)
    return
