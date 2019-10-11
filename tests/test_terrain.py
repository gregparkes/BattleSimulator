#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:17:14 2019

@author: gparkes
"""
import sys
sys.path.insert(0,"../")
import battlesim as bsm

import pytest

def test_define_terrain():
    _ = bsm.Terrain()
    # test with bad settings.
    with pytest.raises(TypeError):
        bsm.Terrain("hello")
    with pytest.raises(TypeError):
        bsm.Terrain(42)
    with pytest.raises(TypeError):
        bsm.Terrain((0, 10, 0, 10), "hello", "contour")
    with pytest.raises(AttributeError):
        bsm.Terrain((0, 10, 0, 10), .1, 42)
    with pytest.raises(AttributeError):
        bsm.Terrain((0, 10, 0), .1, "contour")
    with pytest.raises(TypeError):
        bsm.Terrain((0, 10, 0, 10), -1, "contour")
    with pytest.raises(AttributeError):
        bsm.Terrain((0, -1, 0, 10), .1, "contour")
    with pytest.raises(AttributeError):
        bsm.Terrain((0, 10, 0, -1), .1, "contour")
    with pytest.raises(TypeError):
        bsm.Terrain((0, [0, 1], "hi", 10), .1, "contour")
    with pytest.raises(AttributeError):
        bsm.Terrain((0, 10, 0, 10), .1, "hello contour boy")

    bsm.Terrain((0, 10, 0, 10), .1, None)


def test_terrain_attributes():
    T = bsm.Terrain()

    assert T.Z_ is None, "Z_ should be undefined at this stage"
    assert isinstance(T.dim_, tuple), "dim_ should be a tuple"
    assert len(T.dim_) == 4, "dim_ should be of length 4"
    assert isinstance(T.res_, float), "resolution must be float"
    assert isinstance(T.form_, str), "form_ must be of type str"
    assert T.form_ in [None, "grid", "contour"], "form_ must be None, grid or contour"


def test_generate():
    T = bsm.Terrain()
    # good
    T.generate()

    with pytest.raises(TypeError):
        T.generate("hello")
    with pytest.raises(TypeError):
        T.generate(42)
    # generate with function that does not have 2 parameters

    # generate with function that does not pass 2 numpy arrays
