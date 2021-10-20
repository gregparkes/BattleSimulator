#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes

This class handles the primary simulator functions given some data.
"""
import pandas as pd
import numpy as np

from battlesim import _mathutils

############################################################################

__all__ = ["simulate_battle"]


def frame_columns():
    """Column names w.r.t frame dataframe."""
    return "army", "allegiance", "alive", "armor", "hp", "x", "y", "dir_x", "dir_y"


def _copy_frame(Frames, M, S, i):
    # copy over data from M into frames.
    Frames["frame"][i] = i
    Frames["x"][i] = M["x"]
    Frames["y"][i] = M["y"]
    Frames["target"][i] = M["target"]
    Frames["hp"][i] = np.clip(M["hp"], a_min=0, a_max=None)
    Frames["armor"][i] = np.clip(M["armor"], a_min=0, a_max=None)
    # create direction norm
    dx = M["x"][M['target']] - M['x']
    dy = M["y"][M['target']] - M['y']
    dist = _mathutils.euclidean_distance(dx, dy)
    # now store in two vars ( preventing dist=0 for errors)
    Frames['ddx'][i] = dx / (dist + 1e-12)
    Frames['ddy'][i] = dy / (dist + 1e-12)
    Frames["team"][i] = M["team"]
    Frames["utype"][i] = M["utype"]
    return


def _convert_to_pandas(frames):
    """
    Given the 'frames' from fast_simulate_battle, return a pd.Dataframe
    """
    steps, units = frames.shape
    # use frame as index to reduce memory.
    DF = pd.concat([pd.DataFrame({
        "army": frames["utype"][s],
        "allegiance": frames["team"][s],
        "alive": frames["hp"][s] > 0,
        "hp": np.clip(frames["hp"][s], a_min=0., a_max=None),
        "armor": np.clip(frames["armor"][s], a_min=0, a_max=None),
        "x": frames["x"][s],
        "y": frames["y"][s],
        "dir_x": frames["ddx"][s],
        "dir_y": frames["ddy"][s]
    }, index=frames["frame"][s]) for s in range(steps)])
    DF.index.name = "frame"
    return DF


def simulate_battle(M,
                    S,
                    terrain,
                    target_map,
                    decision_map,
                    max_step=100,
                    ret_frames=True):
    """
    Given a Numpy Matrix of units, simulate a fight.

    This uses a matrix M which is **heterogenous**; that is to say that
    it has named columns [pos, target, hp, dpos, team, group] which
    helps interpretability at the cost of some processing time.

    Columns in M are '("team", np.uint8), ("utype", np.uint8), ("pos", np.float32, 2), ("hp", np.float32),
            ("armor", np.float32), ("range", np.float32), ("speed", np.float32), ("acc", np.float32),
            ("dodge", np.float32), ("dmg", np.float32), ("target", np.int32),
            ("group", np.uint8)'

    Parameters
    --------
    M : np.ndarray (units, )
        A heterogenous matrix containing data values for units
    S : np.ndarray (armies,)
        Static data describing fixed variables.
    terrain : bsm.Terrain object
        Terrain object containing the bounds.
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
    # unpack bounds
    xmin, xmax, ymin, ymax = terrain.bounds_

    # flatten X, Y
    X_m, Y_m = terrain.get_flat_grid()
    # repeat X_m, Y_m for n units.
    X_m = np.repeat(X_m, M.shape[0]).reshape(X_m.shape[0], M.shape[0])
    Y_m = np.repeat(Y_m, M.shape[0]).reshape(Y_m.shape[0], M.shape[0])
    # height
    Z_m = terrain.Z_

    if ret_frames:
        frames = np.zeros(
            (max_step + 1, M.shape[0]),
            dtype=np.dtype([("frame", "u2"), ("x", "f4"), ("y", "f4"), ("target", "u4"),
                            ("hp", "f4"), ("armor", "f4"), ("ddx", "f4"), ("ddy", "f4"),
                            ("team", "u1"), ("utype", "u1")
                            ], align=True)
        )

    while (t < max_step) and running:

        # copy a frame
        if ret_frames:
            _copy_frame(frames, M, S, t)

        # perform a boundary check.
        _mathutils.boundary_check(xmin, xmax, ymin, ymax, M["x"], M['y'])
        # list of indices per unit for which tile they are sitting on (X, Y)
        X_t_ind = np.argmin(np.abs(M["x"] - X_m), axis=0)
        Y_t_ind = np.argmin(np.abs(M["y"] - Y_m), axis=0)

        """# pre-compute the direction derivatives and magnitude/distance for each unit to it's target in batch."""
        dx = M['x'][M['target']] - M['x']
        dy = M['y'][M['target']] - M['y']
        dists = _mathutils.euclidean_distance(dx, dy)
        """precompute enemy and ally target listings"""
        enemy_targets = [np.argwhere((M["hp"] > 0) & (M["team"] != T)).flatten() for T in teams]
        ally_targets = [np.argwhere((M["hp"] > 0) & (M["team"] == T)).flatten() for T in teams]
        """# pre-compute the 'luck' of each unit with random numbers."""
        round_luck = np.random.rand(M.shape[0])

        # iterate over units and check their life, target.
        for i in range(M.shape[0]):
            if M["hp"][i] > 0.:
                # fetch the function 'hit_and_run', 'aggressive' in `_ai.py`, etc.
                dm = decision_map[M["group"][i]]
                # AI-based decision for attack/defend.
                running = (dm(
                    # """The reason why the variables within M are split up is in a previous version,
                    #                     numba did not support this set up of numpy array. It now does!
                    #                     """
                    # the main data matrix.
                    M,
                    # calculated 'luck' rolls for round
                    round_luck,
                    # euclidean distance to targets
                    dists,
                    # directional derivatives to targets
                    dx, dy,
                    # group indices (targets to i, enemies of i, allies of i)
                    target_map[M["group"][i]],
                    enemy_targets[M["team"][i]],
                    ally_targets[M["team"][i]],
                    # variables to do with terrain.
                    Z_m, X_t_ind, Y_t_ind,
                    # current unit under examination - index.
                    i
                ))

        t += 1

    if ret_frames:
        _copy_frame(frames, M, S, t)
        # return _convert_to_pandas(frames[:t])
        return frames[:t]
    else:
        return np.asarray([np.argwhere((M["hp"] > 0) & (M["team"] == T)).flatten().shape[0] for T in teams])
