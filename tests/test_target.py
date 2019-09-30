#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:44:31 2019

@author: gparkes
"""

import sys
import pandas as pd
import numpy as np
import pytest

from scipy.stats import normaltest, kstest

sys.path.insert(0,"../")
import battlesim as bsm


def some_setup():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    # created normally.
    b.create_army([("B1 battledroid", 100), ("Clone Trooper", 100)])
    return b


def get_targets(M):
    teams = np.unique(M["team"])
    enemy_targets = [np.argwhere((M["hp"]>0) & (M["team"]!=T)).flatten() for T in teams]
    ally_targets = [np.argwhere((M["hp"]>0) & (M["team"]==T)).flatten() for T in teams]
    return enemy_targets, ally_targets


def test_random():
    pass


def test_pack():
    pass


def test_nearest():
    pass


def test_weakest():
    pass


def test_strongest():
    pass