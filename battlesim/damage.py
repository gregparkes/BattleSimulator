#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:12:15 2019

@author: gparkes
"""

from numba import jit

@jit(nopython=True)
def basic(hp, target, damage, i):
    hp[target[i]] -= damage[i]


@jit(nopython=True)
def heal(hp, target, damage, i):
    hp[target[i]] += damage[i]