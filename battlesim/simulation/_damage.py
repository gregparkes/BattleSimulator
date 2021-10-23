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

Total Damage (without armor) = Base Damage * ((terra[i] - terra[j]) / 2) + 1
"""


@njit
def bonus_terrain_damage(z_i, z_j):
    """Calculates the bonus terra damage done."""
    return ((z_i - z_j) / 2.) + 1.


@njit
def basic(M, z_i, z_j, i):
    """Determines basic damage output to target i."""
    # cache base damage and target.
    j = M['target'][i]
    _dmg = M['dmg'][i] * (((z_i - z_j) / 2.) + 1.)
    if M['armor'][j] > 0:
        # deal damage to armor (and a little bit to HP)
        diff = M['armor'][j] - _dmg
        if diff < 0:
            M['hp'][j] += diff
        M['armor'][j] -= _dmg
    else:
        M['hp'][j] -= _dmg
