#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:27:46 2019

@author: gparkes
"""
import itertools as it
import os
import pandas as pd


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


def check_groups_in_db(groups, db):
    for group_name, count in groups:
        if group_name not in db.index:
            raise ValueError("group '{}' not found in {}".format(group_name, db.index.tolist()))
    return True


def slice_loop(loopable, n):
    """Returns n elements from an infinite loop of loopable."""
    return list(it.islice(it.cycle(loopable), 0, n))


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


def is_tqdm_installed(raise_error: bool = False):
    """Determines whether tqdm is installed."""
    try:
        from tqdm import tqdm  # noqa
        is_installed = True
    except ModuleNotFoundError:  # pragma: no cover
        is_installed = False
    # Raise error (if needed) :
    if raise_error and not is_installed:  # pragma: no cover
        raise ModuleNotFoundError("tqdm not installed. Use `pip " "install tqdm`.")
    return is_installed
