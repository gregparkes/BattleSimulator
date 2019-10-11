#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:24:02 2019

@author: gparkes
"""

import sys
import pytest

sys.path.insert(0,"../")
import battlesim as bsm


def some_setup():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    # created normally.
    b.create_army([("B1 battledroid", 10), ("Clone Trooper", 10)])
    return b


def test_simulate_battle():
    b = some_setup()
    # call with wrong types
    with pytest.raises(TypeError):
        bsm.simulator_fast.simulate_battle("hello", b.bounds_,
                                           b._rolling_map, b._decision_map, max_step=100)

    # call simulate battle with correct args
    F = bsm.simulator_fast.simulate_battle(b.M_, b.bounds_,
                                           b._rolling_map, b._decision_map, max_step=100)
