#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes
"""
import pandas as pd
import numpy as np
from numba import jit

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


def _convert_to_pandas(frames):
    """
    Given the 'frames' from fast_simulate_battle, return a pd.Dataframe
    """
    steps, units = frames.shape
    DF = pd.concat([pd.DataFrame({
        "frame": frames["frame"][s], "allegiance": frames["team"][s], "alive": frames["hp"][s] > 0,
        "x": frames["pos"][s][:,0], "y": frames["pos"][s][:,1], "dir_x": frames["dpos"][s][:,0],
        "dir_y": frames["dpos"][s][:,1]
    }) for s in range(steps)])
    return DF


@jit(nopython=True)
def _direction_norm(dir_vec):
    mag = np.sqrt(np.sum(dir_vec**2, axis=1))
    return (dir_vec.T/mag).T


def simulate_battle(M, max_step=100, acc_penalty=15., M_threshold=5000):
    """
    Given a Numpy Matrix of units, simulate a fight.

    Parameters
    --------
    M : np.ndarray (units, )
        A matrix containing data values
    max_step : int
        The maximum number of steps
    acc_penalty : float
        Penalties applied to global accuracy
    M_threshold : int
        Threshold before we start calculating global targeting

    Returns
    -------
    frames : np.ndarray (frame, )
        Each frame is a numpy.ndmatrix.
    """
    t = 0
    running = True
    teams = np.unique(M["team"])

    frames = np.zeros(
            (max_step, M.shape[0]),
            dtype=[("frame", int, 1), ("pos", float, 2), ("target", int, 1),
                   ("hp", float, 1), ("dpos", float, 2), ("team", int, 1),
            ]
    )

    def add_frame(M, i):
        # copy over data from M into frames.
        frames["frame"][i] = i
        frames["pos"][i] = M["pos"]
        frames["target"][i] = M["target"]
        frames["hp"][i] = M["hp"]
        # create direction norm
        dnorm = _direction_norm(M["pos"][M["target"]] - M["pos"])
        frames["dpos"][i] = dnorm
        frames["team"][i] = M["team"]
        return

    add_frame(M, 0)

    while (t < max_step) and running:

        group_vec = M["pos"][M["target"]] - M["pos"]
        dists = np.sqrt(np.sum((group_vec)**2, axis=1))

        # if M is large, pre-compute target matrices at each time t rather than on each unit.
        if M.shape[0] > M_threshold:
            valid_targets = [np.argwhere((M["hp"]>0) & (M["team"]!=T)).flatten() for T in teams]

        for i in range(M.shape[0]):
            if M["hp"][i] > 0.:
                if M["hp"][M["target"][i]] <= 0.:
                    # assign new target
                    if M.shape[0]>M_threshold and valid_targets[M["team"][i]].shape[0] > 0:
                        M["target"][i] = np.random.choice(valid_targets[M["team"][i]])
                    else:
                        T = ai.assign_random_target(M, i)
                        if T == -1:
                            running=False
                        else:
                            M["target"][i] = T
                # if not in range, move towards target
                if dists[i] > M["range"][i] or np.random.rand() < .2:
                    # move unit.
                    M["pos"][i] += M["speed"][i] * (group_vec[i] / dists[i])
                else:
                    hit = M["acc"][i] * (1. - M["dodge"][i]) * (1. - dists[i] / acc_penalty)
                    if hit > np.random.rand():
                        M["hp"][M["target"][i]] -= M["dmg"][i]
        t += 1
        add_frame(M, t)

    return _convert_to_pandas(frames[:t])
