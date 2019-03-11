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


def first_sim_without():
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    a1 = bsm.Army(b, "B1 battledroid", 50)
    a2 = bsm.Army(b, "Clone Trooper", 30)
    sim = bsm.simulate_battle([a1, a2], max_timestep=200)
    return sim


def simulation_by_delay():
    b = bsm.Battle("../datasets/starwars-clonewars.csv")
    # is initialised
    b.add(bsm.DelayArmy("B1 battledroid", 50))
    b.add([bsm.DelayArmy("B1 battledroid", 100), bsm.DelayArmy("Clone Trooper",50)])
    # initialsie
    b.instantiate()
    return b


if __name__ == '__main__':
    sim = first_simulation()
    sim2 = first_sim_without()
    #b = simulation_by_delay()
    bsm.quiver_fight(sim)