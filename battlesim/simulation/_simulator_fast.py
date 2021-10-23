#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:48:25 2019

@author: gparkes

This class handles the primary simulator functions given some data.
"""
import numpy as np
from numba import jit, njit, typed

from battlesim.simulation import _ai as AI
from battlesim import _mathutils

############################################################################

__all__ = ["simulate_battle"]


def frame_columns():
    """Column names w.r.t frame dataframe."""
    return "army", "allegiance", "alive", "armor", "hp", "x", "y", "dir_x", "dir_y"


@njit
def _copy_frame(Frames, M, S, dx, dy, dist, i):
    # copy over data from M into frames.
    Frames["x"][i] = M["x"]
    Frames["y"][i] = M["y"]
    Frames["target"][i] = M["target"]
    Frames["hp"][i] = M["hp"]
    Frames["armor"][i] = M["armor"]
    # create direction norm
    # now store in two vars ( preventing dist=0 for errors)
    Frames['ddx'][i] = dx / (dist + 1e-12)
    Frames['ddy'][i] = dy / (dist + 1e-12)
    Frames["team"][i] = M["team"]
    Frames["utype"][i] = M["utype"]
    return


@jit
def _loop_units(M,
                luck,
                dists,
                dx,
                dy,
                enemy_targets,
                ally_targets,
                Z_m):
    """ Loops over the units and executes the function. """
    running = True

    for i in range(M.shape[0]):

        if M['hp'][i] > 0.:
            # fetch the function 'hit_and_run', 'aggressive' in `_ai.py`, etc.
            # AI-based decision for attack/defend.
            k = M['ai_func_index'][i]
            if k == 0:
                running = AI.aggressive(
                    # the main data matrix.
                    M,
                    # calculated 'luck' rolls for round
                    luck,
                    # euclidean distance to targets
                    dists,
                    # directional derivatives to targets
                    dx, dy,
                    # group indices (targets to i, enemies of i, allies of i)
                    enemy_targets[M['team'][i]],
                    ally_targets[M['team'][i]],
                    # variables to do with terrain.
                    Z_m,
                    # current unit under examination - index.
                    i
                )
            elif k == 1:
                running = AI.hit_and_run(
                    # the main data matrix.
                    M,
                    # calculated 'luck' rolls for round
                    luck,
                    # euclidean distance to targets
                    dists,
                    # directional derivatives to targets
                    dx, dy,
                    # group indices (targets to i, enemies of i, allies of i)
                    enemy_targets[M['team'][i]],
                    ally_targets[M['team'][i]],
                    # variables to do with terrain.
                    Z_m,
                    # current unit under examination - index.
                    i
                )
    return running


def simulate_battle(M,
                    S,
                    terrain,
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

    if ret_frames:
        frames = np.zeros(
            (max_step + 1, M.shape[0]),
            dtype=np.dtype([("x", "f4"), ("y", "f4"), ("target", "u4"),
                            ("hp", "f4"), ("armor", "f4"), ("ddx", "f4"), ("ddy", "f4"),
                            ("xtile", "u4"), ("ytile", "u4"), ("team", "u1"), ("utype", "u1")
                            ], align=True)
        )

    #_dec_map = typed.List([AI.aggressive, AI.hit_and_run])

    while (t < max_step) and running:
        """# perform a boundary check."""
        _mathutils.boundary_check(xmin, xmax, ymin, ymax, M["x"], M['y'])
        # lerp to update all units tile position
        M['xtile'] = np.interp(M['x'], [xmin, xmax], [0., terrain.Z_.shape[0]]).astype(np.uint16)
        M['ytile'] = np.interp(M['y'], [ymin, ymax], [0., terrain.Z_.shape[1]]).astype(np.uint16)
        """# pre-compute the direction derivatives and magnitude/distance for each unit to it's target in batch."""
        dx = M['x'][M['target']] - M['x']
        dy = M['y'][M['target']] - M['y']
        dists = _mathutils.euclidean_distance(dx, dy)
        """precompute enemy and ally target listings"""
        enemy_targets = typed.List([np.argwhere((M["hp"] > 0.) & (M["team"] != T)).flatten() for T in teams])
        ally_targets = typed.List([np.argwhere((M["hp"] > 0.) & (M["team"] == T)).flatten() for T in teams])
        """# pre-compute the 'luck' of each unit with random numbers."""
        round_luck = np.random.rand(M.shape[0])

        # copy a frame
        if ret_frames:
            _copy_frame(frames, M, S, dx, dy, dists, t)

        # iterate over units and call AI function.
        running = _loop_units(M, round_luck, dists, dx, dy,
                              enemy_targets,
                              ally_targets,
                              terrain.Z_)

        t += 1

    if ret_frames:
        # return _convert_to_pandas(frames[:t])
        return frames[:t]
    else:
        return np.asarray([np.argwhere((M["hp"] > 0) & (M["team"] == T)).flatten().shape[0] for T in teams])
