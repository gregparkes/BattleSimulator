#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:20:25 2019

@author: gparkes
"""
import sys
import pandas as pd
import numpy as np
import pytest

from scipy.stats import normaltest, kstest

sys.path.insert(0, "../")
import battlesim as bsm


def test_battle_attributes():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    assert b.M_ is None, "M_ should not be set"
    assert type(b.db_) == pd.DataFrame, "db_ must be a pandas.dataframe"
    assert b.sim_ is None, "sim_ not set yet"

    with pytest.raises(AttributeError):
        b.composition_

    return b


def test_battle_create_army():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # create army
    # try normal
    with pytest.raises(TypeError):
        b.create_army("Hello")
    with pytest.raises(TypeError):
        b.create_army(pd.DataFrame({"ho": [1, 2], "hi": [2, 3]}))
    with pytest.raises(TypeError):
        b.create_army(["Clone"])
    with pytest.raises(TypeError):
        b.create_army(["Clone", 2])

    with pytest.raises(ValueError):
        b.create_army([("Clone", 2)])
        b.create_army([("Droid", 10), ("Clone Trooper", 5)])

    with pytest.raises(TypeError):
        b.create_army([("Clone Trooper", "hello")])
        b.create_army([("B1 battledroid", np.inf)])

    # created normally.
    b.create_army([("B1 battledroid", 10), ("Clone Trooper", 10)])
    return b


def test_simulate():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # cannot simulate before creating an army set
    with pytest.raises(AttributeError):
        b.simulate()

    # define army
    b.create_army([("B1 battledroid", 100), ("Clone Trooper", 100)])

    assert b.sim_ is None, "no simulation object present"

    # no important parameters apart from those passed to simulate_battle
    # check return type
    F = b.simulate()
    # check presense of b.sim_
    assert b.sim_ is not None, "simulation object should be present and isnt"
    assert type(F) is np.ndarray, "must be of type np.ndarray for F"
