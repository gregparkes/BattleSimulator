#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:05 2019

@author: gparkes

A selection of algorithms for deciding which enemy to target next.
"""
import numpy as np
from numba import jit


def get_init_function_names():
    return ["random", "pack", "nearest", "weakest", "strongest"]


__all__ = get_init_function_names()


def get_init_functions():
    return [random, pack, nearest, weakest, strongest]

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


def pack(enemies, allies, M, i, k=5):
    """
    Given alive enemy candidates and allies, determine the top k targets
    within a 'pack mentality' i.e all target the same top k enemies. Otherwise
    randomly choose a remaining enemy.

    Parameters
    --------
    enemies : np.ndarray
        indices of alive enemy candidates
    allies : np.ndarray
        indices of alive ally candidates
    M : np.ndarray (n,)
        The total dataset
    i : int
        Index of chosen unit
    k : int
        The number of topk targets

    Returns
    -------
    j : Index of new target
        -1 if not valid target chosen.
    """
    # calculate the intersection of allied targets, with the alive enemies.
    valid_targets = np.intersect1d(M["target"][allies], enemies)
    if enemies.shape[0] > 0:
        if valid_targets.shape[0] > k:
            # return the top k enemies with the most allies targeting it
            topktargets = np.argpartition(np.bincount(valid_targets),k)[:k]
            return np.random.choice(topktargets)
        else:
            # choose a random new enemy
            return np.random.choice(enemies)
    else:
        return -1


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


def weakest(enemies, allies, M, i):
    """
    Given enemy alive candidates, globally determine which one is weakest with
    lowest hit points (and easiest to kill).

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
        return enemies[np.argmin(M["hp"][enemies])]
    else:
        return -1


def strongest(enemies, allies, M, i):
    """
    Given enemy alive candidates, globally determine which one is the strongest
    of the enemies and target them.

    Parameters
    --------
    enemies : np.ndarray
        indices of enemy candidates
    allies : np.ndarray
        indices of ally candidates
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
        return enemies[np.argmax(M["hp"][enemies])]
    else:
        return -1