#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:07:16 2019

@author: gparkes

This file is concerned with movement functions associated with each army group.

There is the default 'move' and fancier options such as A*.
"""

from numba import jit, float32

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
def to_enemy(M, dx, dy, dist, z_i, i):
    """
    Updates M[x,y] towards the target.

    Modifies inplace..

    The update speed is calculated as:
        s_i * (dd_i / m_ij) * (1 - (Z_i / 2))
    """
    # modify dist to prevent it being zero
    D = dist[i] + 1e-12
    # cache the terra + speed influences.
    terrain_tick = (1. - (z_i / 2.)) * M['speed'][i]
    # compute normed directional derivative and update.
    M['x'][i] += (dx[i] / D) * terrain_tick
    M['y'][i] += (dy[i] / D) * terrain_tick


@jit(nopython=True)
def from_enemy(M, dx, dy, dist, z_i, i):
    """
    Updates M[x,y] towards the target.

    Modifies inplace..

    The update speed is calculated as:
        s_i * (dd_i / m_ij) * (1 - (Z_i / 2))
    """
    # modify dist to prevent it being zero
    D = dist[i] + 1e-12
    # cache the terra + speed influences.
    terrain_tick = (1. - (z_i / 2.)) * M['speed'][i]
    # compute normed directional derivative and update.
    M['x'][i] -= (dx[i] / D) * terrain_tick
    M['y'][i] -= (dy[i] / D) * terrain_tick
