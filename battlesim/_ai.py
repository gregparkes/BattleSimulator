#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:07:18 2019

@author: gparkes
"""
from numba import njit

from . import _move
from . import _hit
from . import _damage


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
    pos: (np.ndarray, (n, 2), float all) positions of every unit
    speed: (np.ndarray, (n,) float all) base speed of every unit
    mrange: (np.ndarray, (n,) float all) base range of every unit
    acc: (np.ndarray, (n,) float all) accuracy of every unit
    dodge: (np.ndarray, (n,) float all) dodge of every unit
    target: (np.ndarray, (n,) int all) id of every target
    dmg: (np.ndarray, (n,) float all) base damage of every unit
    hp: (np.ndarray, (n,) float all) current HP of every unit
    armor: (np.ndarray, (n,) float all) current armor of enemy unit
    luck: (np.ndarray, (n,2) float all) luck roll of every unit
    distances: (np.ndarray, (n,) float all) unit distance to target
    dd: (np.ndarray, (n,) float all) directional derivative to target
    team: (np.ndarray, (n, ) int all) team ID of every unit
    target_f: (function) target function.
    enemies: (np.ndarray, (e,) int all) IDs of every enemy unit
    allies: (np.ndarray, (a,) int all) IDs of every allied unit
    Z: (np.ndarray, (res, res) float all) the terrain of the map
    Z_xi: (np.ndarray, (n,) int all) the x-index of Z-terrain tile
    Z_yi: (np.ndarray, (n,) int all) the y-index of Z-terrain tile
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
def _select_enemy(f, pos, targets, hp, enemies, allies, i):
    if hp[targets[i]] <= 0:
        if enemies.shape[0] > 0:
            t = f(pos, hp, enemies, allies, i)
            if t != -1:
                targets[i] = t
                return True
            else:
                return False
        else:
            return False
    else:
        return True


@njit
def _rng_terr(mrange, Z, Z_xi, Z_yi, i):
    """ The range of the unit, modified by the terrain measurement. Returns float. """
    return mrange[i] * _height_modification(Z[Z_xi[i], Z_yi[i]])


""" -------------------------- ACCESSED FUNCTIONS -----------------------------------------"""


@njit
def aggressive(pos, speed, mrange, acc, dodge, targets,
               dmg, hp, armor, luck, distances, dd, team, target_f,
               enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This basic AI looks whether the current unit i is in range of it's target'
    and if it isn't, moves towards it until it is, then attacks.
    """

    """# use ai_map to dictionary-map the group number to the appropriate AI function"""
    """ Arguments: positions, targets, hp, enemies, allies, index, [extras]"""
    if _select_enemy(target_f, pos, targets, hp, enemies, allies, i):
        # if not in range, move towards target, or hit a chance (5%) and move forward anyway.
        if distances[i] > _rng_terr(mrange, Z, Z_xi, Z_yi, i) or luck[i, 1] < 0.05:
            """# move unit towards attacking enemy."""
            pos[i, :] += _move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
            return True
        else:
            """# calculate the chance of hitting the opponent"""
            h_chance = _hit.basic_chance(acc, dodge, distances, i, targets[i])
            """if hit chance overcomes round luck.. deal damage to HP."""
            if h_chance > luck[i, 0]:
                _damage.basic(hp, armor, targets, dmg, Z, Z_xi, Z_yi, i)
            return True
    else:
        return False


@njit
def hit_and_run(pos, speed, mrange, acc, dodge, targets,
                dmg, hp, armor, luck, distances, dd, team, target_f,
                enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This AI option sees if its range/movement is greater than it's enemy, and if it is, it
    performs hit-and-run on it's opponent.
    """
    # assign target enemy
    if _select_enemy(target_f, pos, targets, hp, enemies, allies, i):

        if (speed[i] > speed[targets[i]]) and \
                (_rng_terr(mrange, Z, Z_xi, Z_yi, i) > _rng_terr(mrange, Z, Z_xi, Z_yi, targets[i])):

            # if we're out of range, move towards
            if distances[i] > _rng_terr(mrange, Z, Z_xi, Z_yi, i):
                """# move towards unit."""
                pos[i, :] += _move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
                return True
            # else if the enemy is in range, back off
            elif distances[i] < _rng_terr(mrange, Z, Z_xi, Z_yi, targets[i]):
                """# move directly away from unit."""
                pos[i, :] -= _move.to_enemy(speed, dd, distances, Z, Z_xi, Z_yi, i)
                return True
            else:
                """# so we're in range, the enemy is not, attack."""
                h_chance = _hit.basic_chance(acc, dodge, distances, i, targets[i])
                if h_chance > luck[i, 0]:
                    _damage.basic(hp, armor, targets, dmg, Z, Z_xi, Z_yi, i)
                return True
        else:
            # otherwise just perform an 'aggressive' model.
            return aggressive(pos, speed, mrange, acc, dodge, targets,
                              dmg, hp, armor, luck, distances, dd,
                              team, target_f, enemies, allies,
                              Z, Z_xi, Z_yi, i)
    else:
        return False


def defensive(pos, speed, mrange, acc, dodge, targets,
              dmg, hp, armor, luck, distances, dd, team, target_f,
              enemies, allies, Z, Z_xi, Z_yi, i):
    """
    This AI option attempts to find a nearby high hill and sit on it, waiting for
    a nearby enemy.
    """
    # use target index to see if we can reach the hill in-time.
    return NotImplemented
