#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:41 2019

@author: gparkes
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "../")
import battlesim as bsm


def test_define_distribution():
    # empty no args
    d = bsm.Distribution()

    # wrong type
    with pytest.raises(ValueError):
        bsm.Distribution(1)
    with pytest.raises(TypeError):
        bsm.Distribution([1, 2, 3, "gaussian"])
    # non-option
    with pytest.raises(ValueError):
        bsm.Distribution("hello")

    # working
    bsm.Distribution("normal")
    bsm.Distribution("exp")
    bsm.Distribution("laplace")

    # parameter name not present
    with pytest.raises(ValueError):
        bsm.Distribution("normal", a=0, scale=1.)
    # error with parameters
    with pytest.raises(TypeError):
        bsm.Distribution("normal", loc="same", scale=1.)
    with pytest.raises(TypeError):
        bsm.Distribution("gaussian", loc=1, scale="fifty")

    bsm.Distribution("gaussian", loc=0., scale=1.)


def test_setx():
    # empty no args
    d = bsm.Distribution()

    # set wrong type
    with pytest.raises(TypeError):
        d.setx(1)
    with pytest.raises(TypeError):
        d.setx(loc="hello")
    with pytest.raises(ValueError):
        d.setx(hello=1.)

    d.setx(loc=1.)


def test_sety():
    # empty no args
    d = bsm.Distribution()

    # set wrong type
    with pytest.raises(TypeError):
        d.sety(1)
    with pytest.raises(TypeError):
        d.sety(loc="hello")
    with pytest.raises(ValueError):
        d.sety(hello=1.)

    d.sety(loc=1.)


def test_sample():
    d = bsm.Distribution()

    # set wrong input
    with pytest.raises(TypeError):
        d.sample("some text")
    with pytest.raises(TypeError):
        d.sample(True)

    S = d.sample(10)

    assert isinstance(S, np.ndarray), "S must be an numpy.ndarray"
    assert S.shape[0] == 10, "S must be of set size"
    assert S.shape[1] == 2, "S must be 2D"
