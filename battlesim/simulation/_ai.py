#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:07:18 2019

@author: gparkes
"""
from numba import njit

from . import _damage, _hit, _move


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

"""
Parameters:
    M: heterogeneous np.ndarray (from battle object)
    luck: (np.ndarray, (n,2) float all) luck roll of every unit
    distances: (np.ndarray, (n,) float all) unit distance to target
    dd: (np.ndarray, (n,) float all) directional derivative to target
    target_f: (function) target function.
    enemies: (np.ndarray, (e,) int all) IDs of every enemy unit
    allies: (np.ndarray, (a,) int all) IDs of every allied unit
    Z: (np.ndarray, (res, res) float all) the terra of the map
    Z_xi: (np.ndarray, (n,) int all) the x-index of Z-terra tile
    Z_yi: (np.ndarray, (n,) int all) the y-index of Z-terra tile
    i : (int) the current unit's index
"""


@njit
def _height_modification(h):
    """
    height modifier follows:
        h_{mod} = 1 + (h^2 / 3), h in [0, 1]
    """
    return (h ** 2 / 3.) + 1.


@njit
def _select_enemy(M, f_t, enemies, allies, i):
    if M['hp'][M['target'][i]] <= 0:
        if enemies.shape[0] > 0:
            t = f_t(M, enemies, allies, i)
            if t != -1:
                M['target'][i] = t
                return True
            else:
                return False
        else:
            return False
    else:
        return True


@njit
def _rng_terr(mrange, Z, Z_xi, Z_yi, i):
    """ The range of the unit, modified by the terra measurement. Returns float. """
    return mrange[i] * _height_modification(Z[Z_xi[i], Z_yi[i]])


""" -------------------------- ACCESSED FUNCTIONS -----------------------------------------"""


# pos, speed, mrange, acc, dodge, targets,
#               dmg, hp, armor, luck, distances, dd, team, target_f,
#               enemies, allies, Z, Z_xi, Z_yi, i


@njit
def aggressive(M, luck, dists, dx, dy, target_f,
               enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This basic AI looks whether the current unit i is in range of it's target'
    and if it isn't, moves towards it until it is, then attacks.
    """
    # compute z_i
    zi = Z[Z_xi[i], Z_yi[i]]

    """# use ai_map to dictionary-map the group number to the appropriate AI function"""
    if _select_enemy(M, target_f, enemies, allies, i):
        # if not in range, move towards target, or hit a chance (5%) and move forward anyway.
        if dists[i] > _rng_terr(M['range'], Z, Z_xi, Z_yi, i) or luck[i] < 0.05:
            """# move unit towards attacking enemy."""
            _move.to_enemy(M, dx, dy, dists, Z, Z_xi, Z_yi, i)
            return True
        else:
            """# calculate the chance of hitting the opponent"""
            h_chance = _hit.basic_chance(M, dists, i)
            """if hit chance overcomes round luck.. deal damage to HP."""
            if h_chance > luck[i]:
                _damage.basic(M, Z, Z_xi, Z_yi, i)
            return True
    else:
        return False


@njit
def hit_and_run(M, luck, dists, dx, dy, target_f,
                enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This AI option sees if its range/movement is greater than it's enemy, and if it is, it
    performs hit-and-run on it's opponent.
    """
    # assign target enemy
    if _select_enemy(M, target_f, enemies, allies, i):

        # cache quick stats
        t_i = M["target"][i]
        range_i = _rng_terr(M["range"], Z, Z_xi, Z_yi, i)
        range_j = _rng_terr(M["range"], Z, Z_xi, Z_yi, t_i)

        if (M["speed"][i] > M["speed"][t_i]) and (range_i > range_j):

            # if we're out of range, move towards
            if dists[i] > range_i:
                """# move towards unit."""
                _move.to_enemy(M, dx, dy, dists, Z, Z_xi, Z_yi, i)
                return True
            # else if the enemy is in range, back off
            elif dists[i] < range_j:
                """# move directly away from unit."""
                _move.from_enemy(M, dx, dy, dists, Z, Z_xi, Z_yi, i)
                return True
            else:
                """# so we're in range, the enemy is not, attack."""
                h_chance = _hit.basic_chance(M, dists, i)

                if h_chance > luck[i]:
                    _damage.basic(M, Z, Z_xi, Z_yi, i)
                return True
        else:
            # otherwise just perform an 'aggressive' model.
            return aggressive(M, luck, dists, dx, dy, target_f,
                              enemies, allies, Z, Z_xi, Z_yi, i)
    else:
        return False


def defensive(M, luck, dists, dx, dy, target_f,
                enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This AI option attempts to find a nearby high hill and sit on it, waiting for
    a nearby enemy.
    """
    # use target index to see if we can reach the hill in-time.
    return NotImplemented
