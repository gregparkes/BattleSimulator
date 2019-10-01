#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:24:02 2019

@author: gparkes
"""

import sys
import pandas as pd
import numpy as np
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
        bsm.simulator_fast.simulate_battle("hello", b._mapped_ai, max_step=100, acc_penalty=20.)

    # call simulate battle with correct args
    F = bsm.simulator_fast.simulate_battle(b.M_, b._mapped_ai, max_step=100, acc_penalty=20.)
