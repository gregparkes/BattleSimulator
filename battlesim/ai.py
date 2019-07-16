#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:05 2019

@author: gparkes
"""
import numpy as np
from numba import jit


def get_init_function_names():
    return ["random", "pack", "nearest", "pack_nearest"]


__all__ = get_init_function_names()


def get_init_functions():
    return [random, pack, nearest, pack_nearest]

def get_map_functions():
    return dict(zip(get_init_function_names(), get_init_functions()))

############## AI FUNCTIONS ##############################


@jit(nopython=True)
def random(enemies, allies, M, i):
    """
    Given enemy candidates who are alive, draw one at random.

    Parameters
    --------
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidatea
    M : np.ndarray (n,)
        The total dataset
    i : int
        Index of chosen unit

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.
    """
    # draw a candidate
    if enemies.shape[0] > 0:
        return np.random.choice(enemies)
    else:
        return -1


@jit
def pack(enemies, allies, M, i):
    """
    Given enemy candidates who are alive, assign an enemy based on
    the majority of targets your allies are assigned to.

    Parameters
    --------
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidatea
    M : np.ndarray (n,)
        The total dataset
    i : int
        Index of chosen unit

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.
    """
    # calculate the intersection of allied targets, with the alive enemies.
    valid_targets = np.intersect1d(M["target"][allies], enemies)
    if valid_targets.shape[0] > 0:
        # return the enemy with the most allies targeting it
        return np.argmax(np.bincount(valid_targets))
    else:
        return -1


@jit
def nearest(enemies, allies, M, i):
    """
    Given enemy candidates who are alive, determine which one is nearest.

    Parameters
    --------
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidatea
    M : np.ndarray (n,)
        The total dataset
    i : int
        Index of chosen unit

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.
    """
    if enemies.shape[0] > 0:
        # compute distances/magnitudes
        distances = M["pos"][i] - M["pos"][enemies]
        mags = np.sqrt(np.sum(np.square(distances), axis=1))
        return enemies[np.argmin(mags)]
    else:
        return -1


@jit
def pack_nearest(enemies, allies, M, i, k=5):
    """
    A combination of 'pack' and 'nearest' methods, only consider
    the nearest k allies in pack mentality to select the same
    target.

    Parameters
    --------
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidatea
    M : np.ndarray (n,)
        The total dataset
    i : int
        Index of chosen unit
    k : int
        The number of allies to consider

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.
    """
    if enemies.shape[0] > 0:
        if allies.shape[0] < k:
            ally_indices = allies
        else:
            # select nearest k allies - compute distances
            distances = M["pos"][i] - M["pos"][allies]
            mags = np.sqrt(np.sum(np.square(distances), axis=1))
            # select bottomk allies by magnitude distance
            ally_indices = np.argpartition(mags, k)[:k]
        # calculate valid targets
        valid_targets = np.intersect1d(M["target"][ally_indices], enemies)
        if valid_targets.shape[0] > 0:
            return np.argmax(np.bincount(valid_targets))
        else:
            return -1
    else:
        return -1
