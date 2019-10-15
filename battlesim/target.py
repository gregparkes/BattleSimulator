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

from . import jitcode

def get_init_function_names():
    return ["random", "nearest", "weakest", "strongest", "close_weak"]

def get_global_function_names():
    return ["global_" + n for n in get_init_function_names()]

def get_init_functions():
    return [random, nearest, weakest, strongest, close_weak]

def get_global_functions():
    return [global_random, global_nearest, global_weakest,
            global_strongest, global_close_weak]

def get_map_functions():
    return dict(zip(get_init_function_names(), get_init_functions()))

def get_global_map_functions():
    return dict(zip(get_global_function_names(), get_global_functions()))


__all__ = get_init_function_names() + get_global_function_names()


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
        distances = jitcode.euclidean_distance(pos[i] - pos[enemies])
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
                        (jitcode.remove_bias(hp[enemies]) * (1. - wtc_ratio)) + \
                        (jitcode.remove_bias(jitcode.euclidean_distance(pos[i] - pos[enemies])) * wtc_ratio)
                )
            ]
    else:
        return -1


######################### GLOBAL TARGET ASSIGNMENTS #######################################

"""
A selection of algorithms for deciding all enemies to target.

This is the same as above except there is no index parameter passed. Assumes
all units need a new target.


    Parameters
    --------
    pos : np.ndarray (n, 2)
        The positions of all units
    hp : np.ndarray (n, )
        The HP of every unit.
    team : np.ndarray (n, )
        Team number of every unit.
    group : np.ndarray (n, )
        The group number of every unit.
    group_i : int
        The group number selected

    Returns
    -------
    j : np.ndarray(n, )
        Index(es) of new target
"""


@jit(nopython=True)
def global_random(pos, hp, team, group, group_i):
    # define
    selector = (group == group_i)
    t = np.unique(team[selector])[0]
    # get unit IDs that are not equal to this team for enemies.
    id_not, = np.where(team != t)
    # set the index for these guys
    j = np.random.choice(id_not, selector.sum())
    return j


@jit(nopython=True)
def global_nearest(pos, hp, team, group, group_i):
    # define
    selector = (group == group_i)
    t = np.unique(team[selector])[0]
    # calculate distance matrix, with offset to ignore diagonal, with random noise
    D = jitcode.distance_matrix(pos)
    D += np.eye(D.shape[0])*np.max(D) + np.random.rand(D.shape[0], D.shape[0]) / 4.
    # get unit IDs that are not equal to this team for enemies.
    id_not, = np.where(team != t)
    id_is, = np.where(selector)
    # use distance matrix and ids to select sub groups to find argmin
    j = jitcode.matrix_argmin(D[id_is, :][:, id_not])
    return j


@jit(nopython=True)
def global_weakest(pos, hp, team, group, group_i):
    # define
    selector = (group == group_i)
    t = np.unique(team[selector])[0]
    # get unit IDs that are not equal to this team for enemies.
    id_not, = np.where(team != t)
    id_is, = np.where(selector)

    # find hp of id_not and select weakest.
    j = np.repeat(np.argmin(hp[id_not]), selector.sum())
    return j


@jit(nopython=True)
def global_strongest(pos, hp, team, group, group_i):
    # define
    selector = (group == group_i)
    t = np.unique(team[selector])[0]
    # get unit IDs that are not equal to this team for enemies.
    id_not, = np.where(team != t)
    id_is, = np.where(selector)

    j = np.repeat(np.argmax(hp[id_not]), selector.sum())
    return j


def global_close_weak(pos, hp, team, group, group_i, wtc_ratio=0.7):
    # define
    selector = (group == group_i)
    t = np.unique(team[selector])[0]

    # calculate distance matrix, with offset to ignore diagonal, with random noise
    D = jitcode.distance_matrix(pos)
    D += np.eye(D.shape[0])*np.max(D) + np.random.rand(D.shape[0], D.shape[0]) / 4.

    # return the enemy that is closest and lowest HP
    hp_adj = jitcode.remove_bias(hp) * (1. - wtc_ratio)
    dist_adj = jitcode.remove_bias(D) * wtc_ratio

    # get unit IDs that are not equal to this team for enemies.
    id_not, = np.where(team != t)
    id_is, = np.where(selector)
    # use distance matrix and ids to select sub groups to find argmin
    j = jitcode.matrix_argmin(dist_adj[id_is, :][:, id_not] + hp_adj[id_not])
    return j
