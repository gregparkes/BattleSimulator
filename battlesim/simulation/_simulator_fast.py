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


@njit
def _copy_frame(Frames, M, dx, dy, dist, i):
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


@njit
def _loop_units(M,
                luck,
                dists,
                dx, dy,
                xt, yt,
                enemy_targets,
                Z_m):
    """ Loops over the units and executes the function. """
    running = True

    for i in range(M.shape[0]):

        if M['hp'][i] > 0.:
            # fetch the function 'hit_and_run', 'aggressive' in `_ai.py`, etc.
            # AI-based decision for attack/defend.
            k = M['ai_func_index'][i]
            if k == 0:
                running = AI.aggressive(M, luck, dists, dx, dy, xt, yt, enemy_targets[M['team'][i]], Z_m, i)
            elif k == 1:
                running = AI.hit_and_run(M, luck, dists, dx, dy, xt, yt, enemy_targets[M['team'][i]], Z_m, i)
    return running


@njit
def _step_through_update(M, Z, max_step, teams, enemy_targets, bounds, frames):
    t = 0
    running = True

    zx_index = np.array([0, Z.shape[0]], dtype=np.int64)
    zy_index = np.array([0, Z.shape[1]], dtype=np.int64)
    xb = bounds[:2]
    yb = bounds[2:]

    # begin loop
    while (t < max_step) and running:
        """# perform a boundary check."""
        _mathutils.boundary_check2(bounds, M["x"], M['y'])
        # iterate through and cast every tile element from interpolation.
        xtile = np.interp(M['x'], xb, zx_index)
        ytile = np.interp(M['y'], yb, zy_index)
        """# pre-compute the direction derivatives and magnitude/distance for each unit to it's target in batch."""
        dx = M['x'][M['target']] - M['x']
        dy = M['y'][M['target']] - M['y']
        dists = _mathutils.euclidean_distance(dx, dy)
        """# pre-compute the 'luck' of each unit with random numbers."""
        round_luck = np.random.rand(M.shape[0])
        # loop over enemy target lists and update.
        for g in range(teams.shape[0]):
            # update enemy targets.
            enemy_targets[g] = np.where((M['hp'] > 0.) & (M["team"] != teams[g]))[0]

        """# copy a frame"""
        _copy_frame(frames, M, dx, dy, dists, t)

        """# iterate over units and call AI function."""
        running = _loop_units(M, round_luck, dists, dx, dy, xtile, ytile,
                              enemy_targets, Z)
        t += 1
    return t


@njit
def _step_through_noframe(M, Z, max_step, teams, enemy_targets, bounds):
    """ steps through the simulation. """
    t = 0
    running = True

    zx_index = np.array([0, Z.shape[0]], dtype=np.int64)
    zy_index = np.array([0, Z.shape[1]], dtype=np.int64)
    xb = bounds[:2]
    yb = bounds[2:]

    # begin loop
    while (t < max_step) and running:
        """# perform a boundary check."""
        _mathutils.boundary_check2(bounds, M["x"], M['y'])
        # lerp to update all units tile position
        xtile = np.interp(M['x'], xb, zx_index)
        ytile = np.interp(M['y'], yb, zy_index)
        """# pre-compute the direction derivatives and magnitude/distance for each unit to it's target in batch."""
        dx = M['x'][M['target']] - M['x']
        dy = M['y'][M['target']] - M['y']
        dists = _mathutils.euclidean_distance(dx, dy)
        """# pre-compute the 'luck' of each unit with random numbers."""
        round_luck = np.random.rand(M.shape[0])
        # loop over enemy target lists and update.
        for g in range(teams.shape[0]):
            # update enemy targets.
            enemy_targets[g] = np.where((M['hp'] > 0.) & (M["team"] != teams[g]))[0]
        """# iterate over units and call AI function."""
        running = _loop_units(M, round_luck, dists, dx, dy, xtile, ytile,
                              enemy_targets, Z)
        t += 1
    return t


def simulate_battle(M,
                    terrain,
                    max_step: int = 100,
                    ret_frames: bool = True):
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
    terrain : bsm.Terrain object
        Terrain object containing the bounds.
    max_step : int
        The maximum number of steps
    ret_frames : bool
        If True, save each frame, else, return the allegiance that is victorious.

    Returns
    -------
    frames : np.ndarray (frame, :)
        Each frame is a numpy.ndmatrix.
    """
    # define teams.
    teams = np.unique(M["team"])
    # unpack bounds
    bounds = np.asarray(terrain.bounds_)
    Z = np.copy(terrain.Z_)

    # initialise enemy targets
    enemy_targets = typed.List([np.where((M["team"] != T))[0] for T in teams])

    # initilise frames array if returning full set.
    if ret_frames:
        frames = np.zeros(
            (max_step + 1, M.shape[0]),
            dtype=np.dtype([("x", "f4"), ("y", "f4"), ("target", "u4"),
                            ("hp", "f4"), ("armor", "f4"), ("ddx", "f4"), ("ddy", "f4"),
                            ("team", "u1"), ("utype", "u1")
                            ], align=True)
        )
        t = _step_through_update(M, Z, max_step,
                                 teams, enemy_targets, bounds,
                                 frames)
        return frames[:t]
    else:
        t = _step_through_noframe(M, Z, max_step, teams, enemy_targets,
                                  bounds)
        return np.array([np.sum(np.logical_and(M["hp"] > 0., M["team"] == T)) for T in teams])
