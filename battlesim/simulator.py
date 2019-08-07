#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes
"""
import pandas as pd
import numpy as np
from numba import jit

############################################################################

__all__ = ["simulate_battle", "frame_columns"]


def frame_columns():
    return ["frame", "army", "allegiance", "alive", "x", "y", "dir_x", "dir_y"]


def _convert_to_pandas(frames):
    """
    Given the 'frames' from fast_simulate_battle, return a pd.Dataframe
    """
    steps, units = frames.shape
    DF = pd.concat([pd.DataFrame({
        "frame": frames["frame"][s], "army": frames["group"][s],
        "allegiance": frames["team"][s], "alive": frames["hp"][s] > 0,
        "x": frames["pos"][s][:,0], "y": frames["pos"][s][:,1], "dir_x": frames["dpos"][s][:,0],
        "dir_y": frames["dpos"][s][:,1]
    }) for s in range(steps)])
    return DF


@jit(nopython=True)
def _direction_norm(dir_vec):
    mag = np.sqrt(np.sum(dir_vec**2, axis=1))
    return (dir_vec.T/mag).T


def simulate_battle(M, ai_map, max_step=100, acc_penalty=15., ret_frames=True):
    """
    Given a Numpy Matrix of units, simulate a fight.

    Parameters
    --------
    M : np.ndarray (units, )
        A matrix containing data values
    ai_map : dict
        A dictionary mapping groups (k) to a bsm.ai.* function (v)
    max_step : int
        The maximum number of steps
    acc_penalty : float
        Penalties applied to global accuracy
    ret_frames : bool
        If True, save each frame, else, return the allegiance that is victorious.

    Returns
    -------
    frames : np.ndarray (frame, )
        Each frame is a numpy.ndmatrix.
    """
    t = 0
    running = True
    teams = np.unique(M["team"])

    if ret_frames:
        frames = np.zeros(
                (max_step+1, M.shape[0]),
                dtype=[("frame", np.int64, 1), ("pos", np.float64, 2), ("target", np.int64, 1),
                       ("hp", np.float64, 1), ("dpos", np.float64, 2), ("team", np.uint8, 1),
                       ("group", np.uint8, 1)
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
            frames["group"][i] = M["group"]
            return

        # include the first frame.
        add_frame(M, 0)


    while (t < max_step) and running:

        # pre-compute the direction vectors and magnitudes for each unit in batch.
        group_vec = M["pos"][M["target"]] - M["pos"]
        dists = np.sqrt(np.sum((group_vec)**2, axis=1))
        # pre-compute target matrices at each time t rather than on each unit.
        enemy_targets = [np.argwhere((M["hp"]>0) & (M["team"]!=T)).flatten() for T in teams]
        ally_targets = [np.argwhere((M["hp"]>0) & (M["team"]==T)).flatten() for T in teams]

        # pre-compute the 'luck' of each unit with random numbers.
        round_luck = np.random.rand(M.shape[0])

        # iterate over units and check their life, target.
        for i in range(M.shape[0]):
            if M["hp"][i] > 0.:
                # check whether the target is alive...
                if M["hp"][M["target"][i]] <= 0:
                    # assign new target
                    if enemy_targets[M["team"][i]].shape[0] > 0:
                        # use ai_map to dictionary-map the group number to the appropriate AI function
                        M["target"][i] = ai_map[M["group"][i]](enemy_targets[M["team"][i]], ally_targets[M["team"][i]], M, i)
                    else:
                        running = False

                # if not in range, move towards target
                if dists[i] > M["range"][i]:
                    # move unit towards attacking enemy.
                    M["pos"][i] += M["speed"][i] * (group_vec[i] / dists[i])
                else:
                    # calculate the chance of hitting the opponent
                    # calculated as accuracy * (1 - opponents' dodge) * (1 - distance / accuracy penalty)
                    hit = M["acc"][i] * (1. - M["dodge"][M["target"][i]]) * (1. - dists[i] / acc_penalty)
                    # if hit chance overcomes round luck.. deal damage to HP.
                    if hit > round_luck[i]:
                        M["hp"][M["target"][i]] -= M["dmg"][i]
        t += 1

        if ret_frames:
            add_frame(M, t)

    if ret_frames:
        return _convert_to_pandas(frames[:t])
    else:
        return np.asarray([np.argwhere((M["hp"]>0) & (M["team"]==T)).flatten().shape[0] for T in teams])
