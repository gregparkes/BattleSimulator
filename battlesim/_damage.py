#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:12:15 2019

@author: gparkes
"""

from numba import njit

"""
Includes a bonus damage when attacking downhill.
*New in 0.3.5*: Factoring in Armor
For simplicity, we treat Armor as an extra health-pool for now.

Total Damage (without armor) = Base Damage * ((terrain[i] - terrain[j]) / 2) + 1
"""


@njit
def bonus_terrain_damage(Z, Z_xi, Z_yi, i, j):
    """Calculates the bonus terrain damage done."""
    return ((Z[Z_xi[i], Z_yi[i]] - Z[Z_xi[j], Z_yi[j]]) / 2.) + 1.


@njit
def basic(hp, armor, target, damage, Z, Z_xi, Z_yi, i):
    """Determines basic damage output to target i."""
    # cache base damage.
    dmg = (damage[i] * bonus_terrain_damage(Z, Z_xi, Z_yi, i, target[i]))
    if armor[target[i]] > 0:
        # deal damage to armor (and a little bit to HP)
        diff = armor[target[i]] - dmg
        if diff < 0:
            hp[target[i]] += diff
        armor[target[i]] -= dmg
    else:
        hp[target[i]] -= dmg
