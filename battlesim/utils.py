#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:46 2019

@author: gparkes
"""
import os
import numpy as np
import itertools as it
import pandas as pd


def magnitude(dir_vector):
    """
    Given a numpy.vector of (n,), calculate the magnitude with respect to itself.
    """
    return np.sqrt(np.dot(dir_vector,dir_vector))


def direction_norm(my_unit):
    """
    Returns the normalised direction for the vector given it's and targets position
    """
    v = my_unit.target_.pos_ - my_unit.pos_
    return v / magnitude(v)


def positions_from_spec(pos_dict, N):
    """
    Given the dictionary:
        {"gaussian": {"mean": [0., 0.], "sd": [1., 1.]}}
    or similar from grid, plus N, return N random locations.
    """
    ktype = list(pos_dict.keys())[0]
    if ktype == "gaussian":
        pos_dict["gaussian"]["size"] = (N,2)
        # unpack pos dictionary terms into normal
        pos = np.random.normal(**pos_dict["gaussian"])
    elif ktype == "grid":
        # converts theta (degrees) into radians
        theta = pos_dict["grid"]["theta"] * (np.pi/180.)
        # create rotation matrix
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        factors = np.sort(factor(N))
        nx = factors[len(factors)//2]
        ny = N // nx
        # create a 'grid of points'
        X, Y = grid_of_points(pos_dict["grid"]["xlim"],
                              pos_dict["grid"]["ylim"],
                              nx,
                              ny)
        # stack X, Y together, then rotate with R
        XY = np.vstack((X.reshape(N, order="F"), Y.reshape(N, order="F")))
        pos = np.dot(R, XY).transpose()
    else:
        raise ValueError("ktype '{}' not recognised".format(ktype))
    return pos


def units_from_armies(armies):
    """
    Given a list of Army objects, return a list of Unit objects.
    """
    return list(it.chain.from_iterable([a.units_ for a in armies]))


def distance(u1, u2):
    return np.sqrt(np.dot(u1.pos_ - u2.pos_, u1.pos_ - u2.pos_))


def grid_of_points(xlim, ylim, nx, ny):
    x = np.linspace(xlim[0], xlim[1], nx)
    y = np.linspace(ylim[0], ylim[1], ny)
    return np.meshgrid(x, y)


def factor(n):
    """
    Collect a list of factors given an integer, excluding 1 and n
    """
    def prime_powers(n):
        # c goes through 2, 3, 5 then the infinite (6n+1, 6n+5) series
        for c in it.accumulate(it.chain([2, 1, 2], it.cycle([2,4]))):
            if c*c > n: break
            if n % c: continue
            d, p = (), c
            while not n % c:
                n, p, d = n // c, p * c, d + (p,)
            yield(d)
        if n > 1: yield((n,))

    r = [1]
    for e in prime_powers(n):
        r += [a*b for a in r for b in e]
    return r


def import_and_check_unit_file(fpath):
    """
    Checks the quality of the unit-score datafile.

    Returns dataframe if successful
    """
    # try to import
    if not os.path.isfile(fpath):
        raise IOError("fpath: {} does not exist".format(fpath))
    # attempt to read in
    df = pd.read_csv(fpath)
    cols = df.columns
    must_cols = [
        "Name", "Allegiance", "HP", "Damage", "Accuracy",
        "Miss", "Movement Speed", "Range"
    ]
    for m in must_cols:
        if m not in cols:
            raise IOError("column '{}' not found in file and must be present.".format(m))
    return df
