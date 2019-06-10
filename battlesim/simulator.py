#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes
"""
import pandas as pd
import numpy as np
import itertools as it
import copy

from . import ai
from . import utils

############################################################################

__all__ = ["simulate_battle"]


def extract_frame(units, i):
    # calculate magnitudes for targets
    return pd.DataFrame({
        "frame": i,
        "x": [u.x_ for u in units],
        "y": [u.y_ for u in units],
        "alive": [u.alive_ for u in units],
        "allegiance": [u.allegiance_int_ for u in units],
        "dir_x": [utils.direction_norm(u)[0] for u in units],
        "dir_y": [utils.direction_norm(u)[1] for u in units]
    })


def simulate_battle(armies,
                    max_timestep=100,
                    ret_step=True,
                    acc_penalty=15.,
                    ai_func=ai.ai_random,
                    init_func=ai.init_ai_random):
    """
    Given a list of Army groups - simulate a battle!

    Parameters
    -------
    armies : list of Army
        A set of armies to clash against each other.
    max_timestep : int
        The number max of steps to walk before stopping
    ret_step : bool
        If false, we only return the result of the fight, not stepwise results dataframe
    acc_penalty : float
        The distance penalty to apply to units
    ai_func : function
        The AI program all units will assign targets on
    init_func : function
        The AI program all units initialise assign targets on

    Returns
    -------
    stepwise_simulate : list
        list of pd.DataFrame results.
    """
    # initialise
    t = 0
    running = True
    # collapse the armies down into unit lists.
    units = copy.deepcopy(list(it.chain.from_iterable([a.units_ for a in armies])))

    # initialise enemy unit choice, using init() algorithm
    init_func(units)

    if ret_step:
        stepwise_simulate = []
        # add the first frame with no movement.
        stepwise_simulate.append(extract_frame(units, 0))

    while (t < max_timestep) and running:
        # iterate over every unit and get them to do something.
        for i, unit in enumerate(units):
            # only do something if the unit is alive.
            if unit.alive_:
                # check to see if it's target is alive.
                if not unit.target_.alive_:
                    # assign new target
                    if not ai_func(units, unit):
                        running = False
                # calculate distance from target
                dir_vector = unit.target_.pos_ - unit.pos_
                dist = np.sqrt(np.dot(dir_vector,dir_vector))
                # if not in range, move towards the target
                if dist > unit.range_ or np.random.rand() < 0.2:
                    # adjust position, using normalised velocity vector and movement spped
                    unit.pos_ += unit.move_speed_ * (dir_vector / dist)
                else:
                    # just attack
                    # also factor in distance when calculating hit chance - further away reduces hit
                    hit_chance = unit.accuracy_ * (1. - unit.target_.dodge_) * (1. - dist / acc_penalty)
                    if hit_chance > np.random.rand():
                        unit.target_.curr_hp_ -= unit.damage_

        t += 1
        if ret_step:
            stepwise_simulate.append(extract_frame(units, t))

    if ret_step:
        return pd.concat(stepwise_simulate, sort=False)
    else:
        return {"republic": sum([u.alive_ for u in units if u.allegiance_int_ == 0]),
         "cis": sum([u.alive_ for u in units if u.allegiance_int_ == 1])}
