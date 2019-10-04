#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:34:25 2019

@author: gparkes
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from copy import deepcopy
import os


__units__ = ["Unit1"]
__simulations__ = ["basic_simulation%d" % (i+1) for i in range(3)]
__math__ = ["euclidean", "magnitude", "deriv_norm", "dudt"]
__ai__ = ["find_random_enemy"]
__plot__ = ["plot_quiver"]

# __all__ = __units__ + __simulations__ + __math__ + __ai__ + __plot__
__all__ = []


############################ UNIT CLASSES ############################################

class Unit1(object):
    def __init__(self, db, name):
        """
        Defines a unit, based on examples in the 'db' file, given a name. Assumes the name
        is the index in db.
        """
        # check that name is in db.index
        if not name in db.index:
            raise ValueError("unit {} must be in {}".format(name, db.index))
        self.name = name
        self.hp = db.loc[name, "HP"]
        self.dmg = db.loc[name, "Damage"]
        self.range = db.loc[name, "Range"]
        self.speed = db.loc[name, "Movement Speed"]
        self.team = db.loc[name, "allegiance_int"]
        self.team_name = db.loc[name, "Allegiance"]

        # define position
        self.pos = np.array([0., 0.])
        # directional derivative from target.
        self.dd = np.array([0., 0.])
        # distance from target
        self.dist = 0.
        # define target to aim for
        self.target = None


########################## SIMULATIONS ################################################


def basic_simulation1(units, max_step=100):
    t = 0
    running = True

    # while loop
    while running and t < max_step:
        # iterate over units.
        for i, u in enumerate(units):
            #if we're alive...
            if u.hp > 0:
                # euclidean distance
                u.dd = -deriv_norm(u.pos, u.target.pos)
                u.dist = euclidean(u.pos, u.target.pos)
                # now, if the unit is not in range, then move towards it
                if u.dist > u.range:
                    # speed modifier added, others ignored from above.
                    u.pos += dudt(u.pos, u.target.pos, u.speed)
                else:
                    # perform damage to enemy health.
                    u.target.hp -= u.dmg
        t += 1


def basic_simulation2(units, max_step=50):
    t = 0
    running = True

    steps = []

    # while loop
    while running and t < max_step:
        # iterate over units.
        for i, u in enumerate(units):
            #if we're alive...
            if u.hp > 0:
                # euclidean distance
                u.dd = deriv_norm(u.pos, u.target.pos)
                u.dist = euclidean(u.pos, u.target.pos)
                # now, if the unit is not in range, then move towards it
                if u.dist > u.range:
                    # speed modifier added, others ignored from above.
                    u.pos += dudt(u.pos, u.target.pos, u.speed)
                else:
                    # perform damage to enemy health.
                    u.target.hp -= u.dmg

            # save
            steps.append(deepcopy(units))
        t += 1

    return steps


def basic_simulation3(units, max_step=50):
    t = 0
    running = True

    steps = []

    # while loop
    while running and t < max_step:
        # iterate over units.
        for i, u in enumerate(units):
            #if we're alive...
            if u.hp > 0:

                # wait a second. what if our enemy is dead..?
                if u.target.hp <= 0.:
                    # find a new random enemy.
                    target = find_random_enemy(u, units)
                    if target != -1:
                        u.target = target
                    else:
                        # no more enemies - stop!
                        running = False

                """Only do all this once we have a valid target..."""
                # euclidean distance
                u.dd = deriv_norm(u.pos, u.target.pos)
                u.dist = euclidean(u.pos, u.target.pos)
                # now, if the unit is not in range, then move towards it
                if u.dist > u.range:
                    # speed modifier added, others ignored from above.
                    u.pos += dudt(u.pos, u.target.pos, u.speed)
                else:
                    # perform damage to enemy health.
                    u.target.hp -= u.dmg

            # save
            steps.append(deepcopy(units))
        t += 1

    return steps


################################# EXTRACT FUNCTIONS ###################################


def extract_pos(results, n):
    return np.stack(([np.array([results[i][j].pos for i in range(len(results))]) for j in range(n)]), axis=2)


def extract_dpos(results, n):
    return np.stack(([np.array([results[i][j].dd for i in range(len(results))]) for j in range(n)]), axis=2)


def extract_hp(results, n):
    return np.vstack(([np.array([results[i][j].hp for i in range(len(results))]) for j in range(n)]))


########################## MATH FUNCTIONS ##############################################

def euclidean(u_i, u_j):
    """
    Given two vectors, calculate the distance in 2d space.

    Parameters
    --------
    u_i : np.array(2, )
        2 points x-y unit pos
    u_j : np.array(2, )
        2 points x-y target pos

    Returns
    -------
    d : float
        distance
    """
    return np.sqrt(np.sum(np.square(u_i - u_j)))


def magnitude(p):
    """
    Calculates the magnitude of vector p
    """
    return np.sqrt(np.dot(p, p))


def deriv_norm(u_i, u_j):
    """
    Calculates the normalized directional derivative.

    Parameters
    --------
    u_i : np.array(2, )
        2 points x-y unit pos
    u_j : np.array(2, )
        2 points x-y target pos

    Returns
    -------
    dd : np.array (2, )
        normed directional derivative
    """
    return (u_j - u_i) / euclidean(u_i, u_j)


def dudt(u_i, u_j, s_i):
    """
    Factoring in unit speed, and position points, calculate dudt
    in space and time.
    """
    return s_i * 1. * 1. * deriv_norm(u_i, u_j)


####################### AI FUNCTION #######################################

def find_random_enemy(u, units):
    """Where u is the chosen unit, units is a list of all units."""
    # generate list of alive enemies
    alive_enemies = [i for i in units if (i.hp > 0) and (i.team != u.team)]
    # hang on, what if there are no alive enemies?
    if len(alive_enemies) > 0:
        return random.choice(alive_enemies)
    else:
        return -1


###################### IMPORT ################################


def check_columns(df, list_of_columns):
    for l in list_of_columns:
        if l not in df.columns:
            raise ValueError("column '{}' not found in dataframe.".format(l))


def import_db_file(fpath):
    """
    Imports our dataset of information, given a file path.
    """
    accepted_db_columns = [
        "Name", "Allegiance", "HP", "Damage", "Accuracy",
        "Miss", "Movement Speed", "Range"
    ]

    if not os.path.isfile(fpath):
        raise IOError("fpath does not exist.")
    # attempt to read in
    df = pd.read_csv(fpath)
    # check columns exist
    check_columns(df, accepted_db_columns)
    df.set_index("Name", inplace=True)
    # assign an integer to allegiance
    df["allegiance_int"] = pd.factorize(df["Allegiance"])[0]
    return df


################### PLOTTING FUNCTIONS ################################


def plot_quiver(pos, dpos, legend_cap=5):
    for i in range(pos.shape[-1]):
        if pos.shape[-1] > legend_cap:
            plt.scatter(pos[:, 0, i], pos[:, 1, i])
        else:
            plt.scatter(pos[:, 0, i], pos[:, 1, i], label="unit%d" % (i+1))
        plt.quiver(pos[:, 0, i], pos[:, 1, i], dpos[:, 0, i], dpos[:, 1, i])

    plt.xlabel("x")
    plt.ylabel("y")
    if pos.shape[-1] <= legend_cap:
        plt.legend()
    plt.show()