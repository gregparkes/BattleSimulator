#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:07:18 2019

@author: gparkes
"""
from numba import jit

from . import move
from . import hit
from . import damage


def get_function_names():
    return ["aggressive", "hit_and_run"]

def get_functions():
    return [aggressive, hit_and_run]

def get_function_map():
    return dict(zip(get_function_names(), get_functions()))


__all__ = get_function_names()

"""
Parameters:
    pos: (np.ndarray, (n, 2), float all) positions of every unit
    speed: (np.ndarray, (n,) float all) base speed of every unit
    mrange: (np.ndarray, (n,) float all) base range of every unit
    acc: (np.ndarray, (n,) float all) accuracy of every unit
    dodge: (np.ndarray, (n,) float all) dodge of every unit
    target: (np.ndarray, (n,) int all) id of every target
    dmg: (np.ndarray, (n,) float all) base damage of every unit
    hp: (np.ndarray, (n,) float all) current HP of every unit
    luck: (np.ndarray, (n,2) float all) luck roll of every unit
    distances: (np.ndarray, (n,) float all) unit distance to target
    dd: (np.ndarray, (n,) float all) directional derivative to target
    Z: (np.ndarray, (res, res) float all) the terrain of the map
    Z_xi: (np.ndarray, (n,) int all) the x-index of Z-terrain tile
    Z_yi: (np.ndarray, (n,) int all) the y-index of Z-terrain tile
    i : (int) the current unit's index
"""


@jit(nopython=True)
def aggressive(pos, speed, mrange, acc, dodge, target,
               dmg, hp, luck, distances, dd,
               Z, Z_xi, Z_yi, i):
    """
    This basic AI looks whether the current unit i is in range of it's target'
    and if it isn't, moves towards it until it is, then attacks.
    """
    # if not in range, move towards target, or hit a chance (5%) and move forward anyway.
    if distances[i] > mrange[i] or luck[i, 1] < 0.05:
        """# move unit towards attacking enemy."""
        pos[i, :] += move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
    else:
        """# calculate the chance of hitting the opponent"""
        h_chance = hit.basic_chance(acc, dodge, distances, i, target[i])
        """if hit chance overcomes round luck.. deal damage to HP."""
        if h_chance > luck[i, 0]:
            damage.basic(hp, target, dmg, i)


@jit(nopython=True)
def hit_and_run(pos, speed, mrange, acc, dodge, target,
                dmg, hp, luck, distances, dd,
                Z, Z_xi, Z_yi, i):
    """
    This AI option sees if its range/movement is greater than it's enemy, and if it is, it
    performs hit-and-run on it's opponent.
    """
    if (speed[i] > speed[target[i]]) and (mrange[i] > mrange[target[i]]):
        # if we're out of range, move towards
        if distances[i] > mrange[i]:
            """# move towards unit."""
            pos[i, :] += move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
        # else if the enemy is in range, back off
        elif distances[i] < mrange[target[i]]:
            """# move directly away from unit."""
            pos[i, :] -= move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
        else:
            """# so we're in range, the enemy is not, attack."""
            h_chance = hit.basic_chance(acc, dodge, distances, i, target[i])
            if h_chance > luck[i, 0]:
                damage.basic(hp, target, dmg, i)
    else:
        # otherwise just perform an 'aggressive' model.
        aggressive(pos, speed, mrange, acc, dodge, target,
                   dmg, hp, luck, distances, dd,
                   Z, Z_xi, Z_yi, i)
