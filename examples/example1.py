#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:12:35 2019

@author: gparkes
"""
import time
import sys
sys.path.insert(0, "../")
import battlesim as bsm


def time_func(method, *args):
    ts = time.time()
    result = method(*args)
    te = time.time()
    diff = te - ts
    return diff


def sim1():
    """
    In this first example, we create a 'Battle' class, create a simple army and set gaussian positions.
    """
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    b.create_army([("B1 battledroid",20), ("Clone Trooper",10)])
    b.apply_position_gaussian([(0, 1), (10, 1)])
    # run simulation.
    _ = b.simulate()
    return b


def sim2():
    """
    In this second example, we create a 'Battle' class, create an army and define 'Distributions'
    """
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    b.create_army([("B1 battledroid",20), ("Clone Trooper",10)])
    d1 = bsm.Distribution('normal').setx(loc=5, scale=2)
    d2 = bsm.Distribution('uniform').yoff(4)
    b.apply_position([d1, d2])
    # run
    _ = b.simulate()
    return b


if __name__ == '__main__':
    b1 = sim1()
    b2 = sim2()