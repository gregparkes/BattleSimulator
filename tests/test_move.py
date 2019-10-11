#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:04:50 2019

@author: gparkes
"""

import sys
import numpy as np
sys.path.insert(0,"../")
import battlesim as bsm

def some_setup():
    # battle object requires input file
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    # created normally.
    b.create_army([("B1 battledroid", 10), ("Clone Trooper", 10)])
    return b

