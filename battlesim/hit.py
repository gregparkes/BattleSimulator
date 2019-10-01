#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:20:15 2019

@author: gparkes

Performs calculations on the 'hit chance' of various attacks.
"""

from numba import jit


@jit(nopython=True)
def basic_chance(accuracy, dodge, distances, i, target_i, global_penalty=15.):
    """
    Hit chance in the range [0..1]
    0 meaning no chance of hitting, 1 meaning perfect accuracy.

    Calculated as:

        chance = <accuracy> * (1 - <enemy dodge>) * (1 - <distance to enemy> / <global penalty>)
    """
    return accuracy[i] * (1. - dodge[target_i]) * (1. - distances[i] / global_penalty)
