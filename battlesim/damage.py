#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:12:15 2019

@author: gparkes
"""

from numba import jit


@jit(nopython=True)
def bonus_terrain_damage(Z, Z_xi, Z_yi, i, j):
    return ((Z[Z_xi[i], Z_yi[i]] - Z[Z_xi[j], Z_yi[j]]) / 2.) + 1.


@jit(nopython=True)
def basic(hp, target, damage, Z, Z_xi, Z_yi, i):
    """
    Includes a bonus damage when attacking downhill.

    Total Damage = Base Damage * ((terrain[i] - terrain[j]) / 2) + 1
    """
    hp[target[i]] -= (damage[i] * bonus_terrain_damage(Z,Z_xi,Z_yi,i,target[i]))
