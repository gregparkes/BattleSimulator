#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:12:35 2019

@author: gparkes
"""
import sys
sys.path.insert(0, "../")
import battlesim as bsm


def first_simulation():
    with bsm.Battle("../datasets/starwars-clonewars.csv") as b:
        # where b is a 'Battle' object.
        trial = [
            bsm.Army(b, "B1 battledroid", 50),
            bsm.Army(b, "Clone Trooper", 30)
        ]
        # create simulation
        sim = bsm.simulate_battle(trial, max_timestep=500)

    return sim

if __name__ == '__main__':
    sim = first_simulation()