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


@jit(nopython=True)
def aggressive(pos, speed, mrange, acc, dodge, target, dmg, hp, luck, distances, dd, i):
    """
    This basic AI looks whether the current unit i is in range of it's target'
    and if it isn't, moves towards it until it is, then attacks.
    """
    # if not in range, move towards target, or hit a chance (5%) and move forward anyway.
    if distances[i] > mrange[i] or luck[i, 1] < 0.05:
        """# move unit towards attacking enemy."""
        move.to_enemy(pos, speed, dd, distances, i)
    else:
        """# calculate the chance of hitting the opponent"""
        h_chance = hit.basic_chance(acc, dodge, distances, i, target[i])
        """if hit chance overcomes round luck.. deal damage to HP."""
        if h_chance > luck[i, 0]:
            damage.basic(hp, target, dmg, i)


@jit(nopython=True)
def hit_and_run(pos, speed, mrange, acc, dodge, target, dmg, hp, luck, distances, dd, i):
    """
    This AI option sees if its range/movement is greater than it's enemy, and if it is, it
    performs hit-and-run on it's opponent.
    """
    if (speed[i] > speed[target[i]]) and (mrange[i] > mrange[target[i]]):
        # if we're out of range, move towards
        if distances[i] > mrange[i]:
            """# move towards unit."""
            move.to_enemy(pos, speed, dd, distances, i)
        # else if the enemy is in range, back off
        elif distances[i] < mrange[target[i]]:
            move.from_enemy(pos, speed, dd, distances, i)
        else:
            # so we're in range, the enemy is not, attack.
            h_chance = hit.basic_chance(acc, dodge, distances, i, target[i])
            if h_chance > luck[i, 0]:
                damage.basic(hp, target, dmg, i)
    else:
        # otherwise just perform an 'aggressive' model.
        aggressive(pos, speed, mrange, acc, dodge, target, dmg, hp, luck, distances, dd, i)
