#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:12:35 2019

@author: gparkes
"""
import time
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "../")
import battlesim as bsm


def time_func(method, *args):
    ts = time.time()
    result = method(*args)
    te = time.time()
    diff = te - ts
    return diff


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


def simulate_nbattle(n1, n2):
    print("n1:%d, n2:%d" % (n1,n2))
    with bsm.Battle("../datasets/starwars-clonewars.csv") as b:
        t =[
            bsm.Army(b, "B1 battledroid", n1),
            bsm.Army(b, "Clone Trooper", n2)
        ]
    sim = bsm.simulate_battle(t, max_timestep=n1*n2/10)
    return sim


def simulate_multiple():
    x1 = np.linspace(10, 400, 10, dtype=np.int)
    varx = list(it.combinations(x1, 2))
    times = [time_func(simulate_nbattle, v1, v2) for v1,v2 in varx]
    return np.asarray(varx), np.asarray(times)


if __name__ == '__main__':
    sim = first_simulation()
    sim2 = first_sim_without()
    varx, times = simulate_multiple()
    Nt = varx.sum(axis=1)
    m,b = np.polyfit(Nt, times, 1)
    plt.plot(Nt, times, 'x', Nt, Nt*m+b, 'x')
    plt.xlabel("Total N")
    plt.ylabel("Time taken")