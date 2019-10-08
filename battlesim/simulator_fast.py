#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes

This class handles the primary simulator functions given some data.
"""
import pandas as pd
import numpy as np

from . import utils
from . import ai

############################################################################

__all__ = ["simulate_battle"]


def frame_columns():
    return ["frame", "army", "allegiance", "alive", "x", "y", "dir_x", "dir_y"]


def _copy_frame(Frames, M, i):
    # copy over data from M into frames.
    Frames["frame"][i] = i
    Frames["pos"][i] = M["pos"]
    Frames["target"][i] = M["target"]
    Frames["hp"][i] = M["hp"]
    # create direction norm
    dnorm = utils.direction_norm(M["pos"][M["target"]] - M["pos"])
    Frames["dpos"][i] = dnorm
    Frames["team"][i] = M["team"]
    Frames["utype"][i] = M["utype"]
    return


def _convert_to_pandas(frames):
    """
    Given the 'frames' from fast_simulate_battle, return a pd.Dataframe
    """
    steps, units = frames.shape
    DF = pd.concat([pd.DataFrame({
        "frame": frames["frame"][s], "army": frames["utype"][s],
        "allegiance": frames["team"][s], "alive": frames["hp"][s] > 0,
        "x": frames["pos"][s][:,0], "y": frames["pos"][s][:,1], "dir_x": frames["dpos"][s][:,0],
        "dir_y": frames["dpos"][s][:,1]
    }) for s in range(steps)])
    return DF


def simulate_battle(M,
                    target_map,
                    decision_map,
                    max_step=100,
                    ret_frames=True):
    """
    Given a Numpy Matrix of units, simulate a fight.

    This uses a matrix M which is **heterogenous**; that is to say that
    it has named columns [pos, target, hp, dpos, team, group] which
    helps interpretability at the cost of some processing time.

    Parameters
    --------
    M : np.ndarray (units, )
        A heterogenous matrix containing data values for units
    target_map : dict
        A dictionary mapping groups (k) to a bsm.target.* function (v)
    decision_map : dict
        A dictionary mapping groups (k) to a bsm.ai.* function (v)
    max_step : int
        The maximum number of steps
    ret_frames : bool
        If True, save each frame, else, return the allegiance that is victorious.

    Returns
    -------
    frames : np.ndarray (frame, :)
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
                       ("utype", np.uint8, 1)
                ]
        )
        # include the first frame.
        _copy_frame(frames, M, 0)

    while (t < max_step) and running:

        """# pre-compute the direction derivatives and magnitude/distance for each unit to it's target in batch."""
        dir_vec = M["pos"][M["target"]] - M["pos"]
        dists = utils.euclidean_distance(dir_vec)
        # pre-compute target matrices at each time t rather than on each unit.
        enemy_targets = [np.argwhere((M["hp"]>0) & (M["team"]!=T)).flatten() for T in teams]
        ally_targets = [np.argwhere((M["hp"]>0) & (M["team"]==T)).flatten() for T in teams]

        # pre-compute the 'luck' of each unit with random numbers.
        round_luck = np.random.rand(M.shape[0], 2)

        # iterate over units and check their life, target.
        for i in range(M.shape[0]):
            if M["hp"][i] > 0.:
                # check whether the target is alive...
                if M["hp"][M["target"][i]] <= 0:
                    # assign new target
                    if enemy_targets[M["team"][i]].shape[0] > 0:
                        """# use ai_map to dictionary-map the group number to the appropriate AI function"""
                        """ Arguments: positions, targets, hp, enemies, allies, index, [extras]"""
                        M["target"][i] = target_map[M["group"][i]](
                                M["pos"],
                                M["target"],
                                M["hp"],
                                enemy_targets[M["team"][i]],
                                ally_targets[M["team"][i]],
                                i
                        )
                    else:
                        running = False

                # AI-based decision for attack.
                decision_map[M["team"][i]](
                     # variables
                     M["pos"], M["speed"], M["range"], M["acc"], M["dodge"],
                     M["target"], M["dmg"], M["hp"], round_luck, dists, dir_vec, i
                )

        t += 1

        if ret_frames:
            _copy_frame(frames, M, t)

    if ret_frames:
        return _convert_to_pandas(frames[:t])
    else:
        return np.asarray([np.argwhere((M["hp"]>0) & (M["team"]==T)).flatten().shape[0] for T in teams])
