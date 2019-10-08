#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:05 2019

@author: gparkes

A selection of algorithms for deciding which enemy to target next.

We use a rather ugly way of doing this - so we can use jit to speed up
the calculation using C rather than Python - all the numpy arrays are
passed to the functions.


    Parameters
    --------
    pos : np.ndarray (n, 2)
        The positions of all units
    targets : np.ndarray (n,)
        The target of every unit.
    hp : np.ndarray (n, )
        The HP of every unit.
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidates
    i : int
        Index of chosen unit

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.

"""
import numpy as np
from numba import jit

from . import utils

def get_init_function_names():
    return ["random", "nearest", "weakest", "strongest", "close_weak"]

def get_init_functions():
    return [random, nearest, weakest, strongest, close_weak]

def get_map_functions():
    return dict(zip(get_init_function_names(), get_init_functions()))


__all__ = get_init_function_names()


@jit(nopython=True)
def minmax(x):
    m = np.min(x)
    return (x - m) / (np.max(x) - m)


@jit(nopython=True)
def remove_bias(x):
    return x - np.mean(x)


############## AI FUNCTIONS ##############################


@jit(nopython=True)
def random(pos, targets, hp, enemies, allies, i):
    """
    Given enemy candidates who are alive, draw one at random.
    """
    # draw a candidate
    if enemies.shape[0] > 0:
        return np.random.choice(enemies)
    else:
        return -1


@jit(nopython=True)
def nearest(pos, targets, hp, enemies, allies, i):
    """
    Given enemy candidates who are alive, determine which one is nearest.
    """
    if enemies.shape[0] > 0:
        # compute distances/magnitudes
        distances = utils.euclidean_distance(pos[i] - pos[enemies])
        return enemies[np.argmin(distances)]
    else:
        return -1


@jit(nopython=True)
def weakest(pos, targets, hp, enemies, allies, i):
    """
    Given enemy alive candidates, globally determine which one is weakest with
    lowest hit points (and easiest to kill).
    """

    if enemies.shape[0] > 0:
        return enemies[np.argmin(hp[enemies])]
    else:
        return -1


@jit(nopython=True)
def strongest(pos, targets, hp, enemies, allies, i):
    """
    Given enemy alive candidates, globally determine which one is the strongest
    of the enemies and target them.

    """
    if enemies.shape[0] > 0:
        return enemies[np.argmax(hp[enemies])]
    else:
        return -1


@jit(nopython=True)
def close_weak(pos, targets, hp, enemies, allies, i, wtc_ratio=0.7):
    """
    Given enemy alive candidates, globally determine which one is the weakest
    and closest, using appropriate weighting for each option.

    Given enemy alive candidates, globally determine which one is the strongest
    of the enemies and target them.

    Parameters (extra)
    --------
    wtc_ratio : float [0..1]
        weak-to-close ratio to determine weighting of each part. Values closer
        to 1 prefer closer enemies, whereas values closer to 0 prefer weaker enemies
    """
    if enemies.shape[0] > 0:
        return enemies[
                np.argmin(
                        (remove_bias(hp[enemies]) * (1. - wtc_ratio)) + \
                        (remove_bias(utils.euclidean_distance(pos[i] - pos[enemies])) * wtc_ratio)
                )
            ]
    else:
        return -1
