#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:46 2019

@author: gparkes
"""
import os
import itertools as it
import pandas as pd


__all__ = ["factor", "import_and_check_unit_file"]


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


def get_segments(army_set):
    """
    Where army_set[i][0] is the unit type, army_set[i][1] is the N, (number of units)
    """
    s = [(0, army_set[0][1])]
    total = army_set[0][1]
    for i in range(1,len(army_set)):
        s.append((total,total+army_set[i][1]))
        total+=army_set[i][1]
    return s


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

    # assign index
    df.set_index("Name", inplace=True)
    # assign int allegiance
    df["allegiance_int"] = pd.factorize(df["Allegiance"])[0]
    return df
