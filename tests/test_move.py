#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:04:50 2019

@author: gparkes
"""

import sys
import numpy as np
sys.path.insert(0,"../")
import battlesim as bsm

def some_setup():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    # created normally.
    b.create_army([("B1 battledroid", 10), ("Clone Trooper", 10)])
    return b


def test_default():
    # battle object requires input file
    b = some_setup()

    direction_f = lambda M: M["pos"][M["target"]] - M["pos"]
    mag_f = lambda direction: np.sqrt(np.sum(direction**2, axis=1))

    Dir = direction_f(b.M_)
    Mag = mag_f(Dir)

    # movement function
    test_move_f = lambda speed, direction, magnitude, i: speed[i] * (direction[i] / magnitude[i])

    print(test_move_f(b.M_["speed"], Dir, Mag, 0))

    nudge_unit0 = test_move_f(b.M_["speed"], Dir, Mag, 0)

    # now actual function
    prev_pos = np.copy(b.M_["pos"])

    bsm.move.default(b.M_["pos"], b.M_["speed"], Dir, Mag, 0, 1.)

    assert np.sum(np.abs(b.M_["pos"]-prev_pos - nudge_unit0)[0,:]) < 1e-8, "distances not calculated the same!"
