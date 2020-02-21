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
import functools
import warnings
import time
from typing import Tuple, Iterable


def check_columns(df, list_of_columns):
    """Check that list of columns elements are not in df."""
    for l in list_of_columns:
        if l not in df.columns:
            raise ValueError("column '{}' not found in dataframe.".format(l))


def check_list_type(L, t):
    """Checks that every element in L is of type t."""
    for i, l in enumerate(L):
        if not isinstance(l, t):
            raise TypeError("type '{}' not found in list at index [{}]".format(t, i))
    return True


def colorwheel() -> Tuple:
    """All of the supported named-types in Matplotlib."""
    return ("red", "blue", "green", "orange", "purple", "brown", "black",
            "cyan", "yellow")


def check_in_list(L, sl: Iterable):
    """
    L is the reference list, sl is the 'sub list'
    """
    for i, l in enumerate(sl):
        if l not in L:
            raise ValueError("element '{}' not found in super-list {}".format(l, L))
    return True


def time_n(method, r=5, *args, **kws):
    """
    r for repeats
    """
    T = np.zeros(r + 1, )
    for i in range(r + 1):
        ts = time.time()
        _ = method(*args, **kws)
        T[i] = time.time() - ts
    # drop the first example as it is always slowest
    return T[1:]


def timed(method, *args, **kw):
    """Time a particular function."""
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    diff = te - ts
    """
    Ranges : 0 to 80 (seconds)
    81 to 4800 (minutes)
    4801 to 100800 (hours)
    100800 + (days)
    """
    if diff < 80:
        out_str = "%r: %2.2f s"
    elif 80 <= diff < 4800:
        out_str = "%r: %2.2f m"
        diff /= 60.
    elif 4800 <= diff < 100800:
        out_str = "%r: %2.2f h"
        diff /= 3600.
    else:
        out_str = "%r: %2.2f d"
        diff /= 86400.

    print(out_str % (method.__name__, diff))
    return result


def timeit(method):
    """A decorator for timing methods."""
    def _timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        diff = te - ts
        """
        Ranges : 0 to 80 (seconds)
        81 to 4800 (minutes)
        4801 to 100800 (hours)
        100800 + (days)
        """
        if diff < 80:
            out_str = "%r: %2.2f s"
        elif 80 <= diff < 4800:
            out_str = "%r: %2.2f m"
            diff /= 60.
        elif 4800 <= diff < 100800:
            out_str = "%r: %2.2f h"
            diff /= 3600.
        else:
            out_str = "%r: %2.2f d"
            diff /= 86400.

        print(out_str % (method.__name__, diff))
        return result

    return _timed


def is_twotuple(L, type1, type2):
    """
    Checks whether the object L is a 2-element tuple throughout.
    """
    if isinstance(L, (list, tuple)):
        for i, x in enumerate(L):
            if isinstance(x, tuple):
                if len(x) != 2:
                    raise ValueError("'L' element at index [{}] is not of size 2".format(i))
                if not isinstance(x[0], type1):
                    raise TypeError("x[0] element at index [{}] is not of type '{}'".format(i, type1))
                if not isinstance(x[1], type2):
                    raise TypeError("x[1] element at index [{}] is not of type '{}'".format(i, type2))
            else:
                raise TypeError("'x' is not of type 'tuple'")
        return True
    else:
        raise TypeError("'L' must be of type 'list, tuple'")


def is_ntuple(L, *types):
    """
    Checks whether the object L is an n-element tuple throughout.
    """
    if isinstance(L, (list, tuple)):
        if len(L) == len(types):
            for elem, t in zip(L, types):
                if not isinstance(elem, t):
                    raise TypeError("'L' element {} is not of type {}".format(elem, t))
        else:
            raise ValueError("L must be the same length as type list")
    else:
        raise TypeError("'L' must be of type [list, tuple]")


def get_segments(army_set: Tuple[str, int]):
    """
    Where army_set[i][0] is the unit type, army_set[i][1] is the N, (number of units)
    """
    s = [(0, army_set[0][1])]
    total = army_set[0][1]
    for i in range(1, len(army_set)):
        s.append((total, total + army_set[i][1]))
        total += army_set[i][1]
    return s


def check_groups_in_db(groups, db):
    for group_name, count in groups:
        if group_name not in db.index:
            raise ValueError("group '{}' not found in {}".format(group_name, db.index.tolist()))
    return True


def slice_loop(loopable, n):
    """Returns n elements from an infinite loop of loopable."""
    return list(it.islice(it.cycle(loopable), 0, n))


def max_norm(x):
    """Assumes x is a vector"""
    return x / np.max(x)


def sum_norm(x):
    return x / np.sum(x)


def io_table_columns():
    """Accepted column names for a unit file."""
    return (
        "Name", "Allegiance", "HP", "Armor", "Damage", "Accuracy",
        "Miss", "Movement Speed", "Range"
    )


def io_table_descriptions():
    """Descriptions corresponding to each accepted column name for a unit file."""
    return (
        "The name of the unit. Format string",
        "The team/allegiance of the unit. Format string, must be hashable",
        "HP: the health of the unit; an integer or float, no limit. Must be > 0",
        "Armor: the armor of the unit; an integer or float, no limit. Must be > 0",
        "Damage: the primary damage of the unit; integer or float, no limit.",
        "Accuracy: the accuracy of the unit; an integer/float in the range [0, 100]",
        "Miss: the chance of the unit to miss an attack; an integer/float in the range [0, 100]",
        "Movement Speed: the movement speed of the unit; float.",
        "Range: the range of the unit; integer or float, Must be > 0"
    )


def check_unit_file(df):
    """ Works inplace; no return """
    mappp = dict(zip(io_table_columns(), io_table_descriptions()))
    for m in io_table_columns():
        if m not in df.columns:
            raise IOError("column '{}' not found in file and must be present. Description:::'{}'".format(m, mappp[m]))


def preprocess_unit_file(df):
    """ Works inplace; no return """
    # assign index
    df.set_index("Name", inplace=True)
    # assign int allegiance
    df["allegiance_int"] = pd.factorize(df["Allegiance"])[0]


def import_and_check_unit_file(fpath):
    """
    Checks the quality of the unit-score datafile.

    Returns dataframe if successful.
    """
    # try to import
    if not os.path.isfile(fpath):
        raise IOError("fpath: {} does not exist".format(fpath))
    # attempt to read in
    df = pd.read_csv(fpath)
    # check
    check_unit_file(df)
    # preprocess
    preprocess_unit_file(df)

    return df
