#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:20:25 2019

@author: gparkes
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# import local package
import battlesim as bsm

ROOT = (Path(__file__).parent / "../").resolve()


@pytest.fixture
def battle() -> bsm.Battle:
    """Generates a battle object."""
    return bsm.Battle(str(ROOT / "datasets/starwars-clonewars.csv"))


def test_battle_attributes(battle: bsm.Battle):
    """Tests basic attributes."""
    # battle object requires input file
    assert battle.M_ is None, "M_ should not be set"
    assert type(battle.db_) == pd.DataFrame, "db_ must be a pandas.dataframe"
    assert battle.sim_ is None, "sim_ not set yet"


def test_battle_create_army(battle: bsm.Battle):
    # battle object requires input file
    # try normal
    with pytest.raises(TypeError):
        battle.create_army("Hello")
    with pytest.raises(TypeError):
        battle.create_army(pd.DataFrame({"ho": [1, 2], "hi": [2, 3]}))
    with pytest.raises(TypeError):
        battle.create_army(["Clone"])
    with pytest.raises(TypeError):
        battle.create_army(["Clone", 2])

    with pytest.raises(TypeError):
        battle.create_army([("Clone", 2)])
        battle.create_army([("Droid", 10), ("Clone Trooper", 5)])

    with pytest.raises(TypeError):
        battle.create_army([("Clone Trooper", "hello")])
        battle.create_army([("B1 battledroid", np.inf)])

    # created normally.
    comp = [bsm.Composite("B1 battledroid", 10), bsm.Composite("Clone Trooper", 10)]
    battle.create_army(comp)


def test_simulate(battle: bsm.Battle):
    # cannot simulate before creating an army set
    with pytest.raises(AttributeError):
        battle.simulate()

    # create a composite
    comp = [bsm.Composite("B1 battledroid", 100), bsm.Composite("Clone Trooper", 100)]
    # define army
    battle.create_army(comp)

    assert battle.sim_ is None, "no simulation object present"

    # no important parameters apart from those passed to simulate_battle
    # check return type
    F = battle.simulate()
    # check presense of b.sim_
    assert battle.sim_ is not None, "simulation object should be present and isnt"
    assert type(F) is np.ndarray, "must be of type np.ndarray for F"
