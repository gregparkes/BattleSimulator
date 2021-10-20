#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:20:15 2019

@author: gparkes

Performs calculations on the 'hit chance' of various attacks.
"""

from numba import njit


@njit
def basic_chance(M, dist, i, global_penalty=15.):
    """
    Hit chance in the range [0..1]
    0 meaning no chance of hitting, 1 meaning perfect accuracy.

    Calculated as:
        chance = <accuracy> * (1 - <enemy dodge>) * (1 - <distance to enemy> / <global penalty>)
    """
    return M['acc'][i] * (1. - M['dodge'][M['target'][i]]) * (1. - dist[i] / global_penalty)
