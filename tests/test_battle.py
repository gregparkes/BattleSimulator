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

sys.path.insert(0,"../")
import battlesim as bsm


def test_define_battle():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")


def test_battle_attributes():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    assert b.M_ is None, "M_ should not be set"
    assert type(b.db_) == pd.DataFrame, "db_ must be a pandas.dataframe"
    assert isinstance(b.db_names_, (list, tuple)), "db_names_ must be a list, tuple"
    assert b.sim_ is None, "sim_ not set yet"

    with pytest.raises(AttributeError):
        b.composition_
    with pytest.raises(AttributeError):
        b.n_allegiance_

    return b


def test_battle_create_army():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # create army
    # try normal
    with pytest.raises(AssertionError):
        b.create_army("Hello")
    with pytest.raises(AssertionError):
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


def test_apply_position():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # first, attempt to call without create_army
    with pytest.raises(AttributeError):
        b.apply_position("gaussian")

    # created normally.
    b.create_army([("B1 battledroid", 100), ("Clone Trooper", 100)])

    # try parameter diffs
    with pytest.raises(TypeError):
        b.apply_position()
    with pytest.raises(TypeError):
        b.apply_position(1)

    # apply_pos with false distribution
    with pytest.raises(ValueError):
        b.apply_position("fake")

    # normal
    b.apply_position("gaussian")
    # check distribution - pvalue > 0.05
    assert normaltest(b.M_["pos"][:, 0])[1] > 0.05, "P-value for gaussian test > 0.05, fail"
    b.apply_position("exp")
    # check distribution using KS-test > 0.05
    assert kstest(b.M_["pos"][:, 0], "expon")[1] > 0.05, "P-value for KS-test not > 0.05, fail"
    b.apply_position("laplace")
    assert kstest(b.M_["pos"][:, 1], "laplace")[1] > 0.05, "P-value for KS-test not > 0.05, fail"

    #
    with pytest.raises(ValueError):
        b.apply_position(["fake"])
        b.apply_position([("Hello")])

    # dist for each unit - works
    b.apply_position(["gaussian","uniform"])
    assert kstest(b.M_["pos"][:100, 0], "norm")[1] > 0.01, "P-value for KS-test not > 0.05, fail"
    assert kstest(b.M_["pos"][-100:, 0], "uniform")[1] > 0.01, "P-value for KS-test not > 0.05, fail"

    # we don't test bsm.Distribution objects here
    b.apply_position({"name":"gaussian", "loc": 0., "scale": 1.})

    with pytest.raises(ValueError):
        #wrong index
        b.apply_position({"n":"gaussian", "loc":0., "scale":1.})
    with pytest.raises(TypeError):
        b.apply_position({"name":10, "loc":0., "scale":1.})
    with pytest.raises(ValueError):
        b.apply_position({"name":str(10), "loc":0., "scale":1.})
    with pytest.raises(TypeError):
        # wrong loc, scale parameters
        b.apply_position({"name":"gaussian", "loc":"hello", "scale":1.})


def test_set_initial_ai():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # set AI before definition
    with pytest.raises(AttributeError):
        b.set_initial_ai(["nearest", "pack"])
    with pytest.raises(AttributeError):
        b.set_initial_ai("nearest")

    # created normally.
    b.create_army([("B1 battledroid", 100), ("Clone Trooper", 100)])

    # call error
    with pytest.raises(ValueError):
        b.set_initial_ai("hello")
    with pytest.raises(ValueError):
        b.set_initial_ai(["nearest","hello"])

    # call normally
    b.set_initial_ai(["nearest", "random"])
    # check composition
    assert set(b.composition_["init_ai"].values) == set(["random","nearest"])


def test_set_rolling_ai():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")

    # set AI before definition
    with pytest.raises(AttributeError):
        b.set_rolling_ai(["nearest", "random"])
    with pytest.raises(AttributeError):
        b.set_rolling_ai("nearest")

    # created normally.
    b.create_army([("B1 battledroid", 100), ("Clone Trooper", 100)])

    # call error
    with pytest.raises(ValueError):
        b.set_rolling_ai("hello")
    with pytest.raises(ValueError):
        b.set_rolling_ai(["nearest","hello"])

    # call normally
    b.set_rolling_ai(["nearest", "random"])
    # check composition
    assert set(b.composition_["rolling_ai"].values) == set(["random","nearest"])


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
    assert type(F) is pd.DataFrame, "must be of type pd.DataFrame for F"

    # check columns
    for c in F.columns:
        assert c in bsm.simulator_fast.frame_columns(), "column '{}' not in df.columnset".format(c)
