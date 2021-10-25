#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:07:18 2019

@author: gparkes
"""
import math
from numba import jit, njit

from . import _damage, _hit, _move
from ._target import nearest


def get_function_names():
    """Returns the function names."""
    return "aggressive", "hit_and_run"


def get_functions():
    """Returns the functions themselves."""
    return aggressive, hit_and_run


def get_map_functions():
    """Maps the function names to the functions."""
    return dict(zip(get_function_names(), get_functions()))


__all__ = get_function_names()


@njit
def _select_enemy(M, enemies, i):
    if M['hp'][M['target'][i]] <= 0:
        if enemies.shape[0] > 0:
            t = nearest(M, enemies, i)
            if t != -1:
                M['target'][i] = t
                return True
            else:
                return False
        else:
            return False
    else:
        return True


""" -------------------------- ACCESSED FUNCTIONS -----------------------------------------"""


# pos, speed, mrange, acc, dodge, targets,
#               dmg, hp, armor, luck, distances, dd, team, target_f,
#               enemies, allies, Z, Z_xi, Z_yi, i


@njit
def aggressive(M, luck, dists, dx, dy, xt, yt, enemies, Z, i):
    """
    This basic AI looks whether the current unit i is in range of it's target'
    and if it isn't, moves towards it until it is, then attacks.
    """
    # fetch height for unit i given indices.
    j = M['target'][i]
    z_i = Z[math.trunc(xt[i]), math.trunc(yt[i])]
    z_j = Z[math.trunc(xt[j]), math.trunc(yt[j])]
    # calculate updated range of unit
    r_i = M['range'][i] * ((z_i ** 2 / 3.) + 1.)

    """# use ai_map to dictionary-map the group number to the appropriate AI function"""
    if _select_enemy(M, enemies, i):
        # if not in range, move towards target, or hit a chance (5%) and move forward anyway.
        if dists[i] > r_i or luck[i] < 0.05:
            """# move unit towards attacking enemy."""
            _move.to_enemy(M, dx, dy, dists, z_i, i)
            return True
        else:
            """# calculate the chance of hitting the opponent"""
            h_chance = _hit.basic_chance(M, dists, i)
            """if hit chance overcomes round luck.. deal damage to HP."""
            if h_chance > luck[i]:
                _damage.basic(M, z_i, z_j, i)
            return True
    else:
        return False


@njit
def hit_and_run(M, luck, dists, dx, dy, xt, yt, enemies, Z, i):
    """
    This AI option sees if its range/movement is greater than it's enemy, and if it is, it
    performs hit-and-run on it's opponent.
    """
    # assign target enemy
    if _select_enemy(M, enemies, i):

        # cache quick stats
        j = M["target"][i]
        z_i = Z[math.trunc(xt[i]), math.trunc(yt[i])]
        z_j = Z[math.trunc(xt[j]), math.trunc(yt[j])]
        range_i = M['range'][i] * ((z_i * z_i) / 3.) + 1.
        range_j = M['range'][j] * ((z_j * z_j) / 3.) + 1.

        if (M["speed"][i] > M["speed"][j]) and (range_i > range_j):

            # if we're out of range, move towards
            if dists[i] > range_i:
                """# move towards unit."""
                _move.to_enemy(M, dx, dy, dists, z_i, i)
                return True
            # else if the enemy is in range, back off
            elif dists[i] < range_j:
                """# move directly away from unit."""
                _move.from_enemy(M, dx, dy, dists, z_i, i)
                return True
            else:
                """# so we're in range, the enemy is not, attack."""
                h_chance = _hit.basic_chance(M, dists, i)

                if h_chance > luck[i]:
                    _damage.basic(M, z_i, z_j, i)
                return True
        else:
            # otherwise just perform an 'aggressive' model.
            return aggressive(M, luck, dists, dx, dy,
                              xt, yt, enemies, Z, i)
    else:
        return False


def defensive(M, luck, dists, dx, dy, enemies, allies, Z, i):
    """
    This AI option attempts to find a nearby high hill and sit on it, waiting for
    a nearby enemy.
    """
    # use target index to see if we can reach the hill in-time.
    return NotImplemented
