#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:34:25 2019

@author: gparkes
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pandas as pd
import itertools as it
import random
import time
from copy import deepcopy
import os


__units__ = ["Unit1"]
__simulations__ = ["basic_simulation%d" % (i+1) for i in range(4)]
__animate__ = ["basic_animate%d" % (i+1) for i in range(3)]
__math__ = ["euclidean", "magnitude", "deriv_norm", "dudt"]
__ai__ = ["find_random_enemy", "find_nearest_enemy"]
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
    while running and (t < max_step):
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
        t += 1


def basic_simulation2(units, max_step=50):
    t = 0
    running = True

    steps = []

    # while loop
    while running and (t < max_step):
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
    # create a copy to prevent overriding
    units = deepcopy(units)
    t = 0
    running = True

    steps = []

    # while loop
    while running and (t < max_step):
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
        # add step
        t += 1

    return steps


def basic_simulation4(units, max_step=50):
    # create a copy of the units
    units = deepcopy(units)
    # begin t
    t = 0
    running = True

    steps = []

    # while loop
    while running and (t < max_step):
        # iterate over units.
        for i, u in enumerate(units):
            #if we're alive...
            if u.hp > 0:

                # wait a second. what if our enemy is dead..?
                if u.target.hp <= 0.:
                    # find a new random enemy.
                    target = find_nearest_enemy(u, units)
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
        # add step
        t += 1

    return steps


############################## ANIMATE FUNCTIONS #########################################

def set_boundary(ax, pos):
    xmin, xmax = np.min(pos[:, 0, :])-1., np.max(pos[:, 0, :])+1.
    ymin, ymax = np.min(pos[:, 1, :])-1., np.max(pos[:, 1, :])+1.
    # set bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def basic_animate1(results, n):

    # extract data points
    pos = extract_pos(results, n)
    plt.rcParams["animation.html"] = "html5"
    N_f = len(results)

    # define a figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # define boundaries - we use a function
    set_boundary(ax, pos)

    # set the object to draw to
    u_points, = ax.plot([], [], 'x', color="black", alpha=.8, markersize=5.)

    fig.tight_layout()
    # close the plot
    plt.close()

    # define init
    def init():
        u_points.set_data(pos[0, 0, :], pos[0, 1, :])
        return u_points

    # define animate
    def animate(i):
        u_points.set_data(pos[i, 0, :], pos[i, 1, :])
        return u_points

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames=N_f, blit=False)
    plt.show()
    return ani


def basic_animate2(results, n):

    N_f = len(results)
    # extract data points
    pos = extract_pos(results, n)
    hp = extract_hp(results, n)
    T = extract_team(results, n)

    T_uniq = np.unique(T)
    plt.rcParams["animation.html"] = "html5"
    # define a figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # define boundaries - we use a function
    set_boundary(ax, pos)

    # create a colormap to map our team to a color - use the colorwheel
    cm = dict(zip(T_uniq, colorwheel()))

    # set the objects to draw to - we set them into lists.
    points_alive = toflat(
        [ax.plot([], [], 'o', color=cm[a], alpha=.6, markersize=10.) for a in T_uniq])
    points_dead = toflat(
        [ax.plot([], [], 'x', color=cm[a], alpha=.4, markersize=5.) for a in T_uniq])

    fig.tight_layout()
    # close the plot
    plt.close()

    # define init
    def init():
        for j, a in enumerate(T_uniq):
            T_Alive = np.argwhere((hp[:, 0] > 0) & (T[:, 0]==a)).flatten()
            T_Dead = np.argwhere((hp[:, 0] <= 0) & (T[:, 0]==a)).flatten()

            if len(T_Alive) > 0:
                points_alive[j].set_data(pos[0, 0, T_Alive], pos[0, 1, T_Alive])
            if len(T_Dead) > 0:
                points_dead[j].set_data(pos[0, 0, T_Dead], pos[0, 1, T_Dead])
        return ((*points_alive, *points_dead))

    # define animate
    def animate(i):
        for j, a in enumerate(T_uniq):
            T_Alive = np.argwhere((hp[:, i] > 0) & (T[:, i]==a)).flatten()
            T_Dead = np.argwhere((hp[:, i] <= 0) & (T[:, i]==a)).flatten()

            if len(T_Alive) > 0:
                points_alive[j].set_data(pos[i, 0, T_Alive], pos[i, 1, T_Alive])
            if len(T_Dead) > 0:
                points_dead[j].set_data(pos[i, 0, T_Dead], pos[i, 1, T_Dead])

        return ((*points_alive, *points_dead))

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames=N_f, blit=False)
    plt.show()
    return ani


def basic_animate3(results, n):

    N_f = len(results)
    # extract data points
    pos = extract_pos(results, n)
    dpos = extract_dpos(results, n)
    hp = extract_hp(results, n)
    T = extract_team(results, n)

    T_uniq = np.unique(T)

    plt.rcParams["animation.html"] = "html5"


    # define a figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # define boundaries - we use a function
    set_boundary(ax, pos)

    # create a colormap to map our team to a color - use the colorwheel
    cm = dict(zip(T_uniq, colorwheel()))


    # set the objects to draw to - we set them into lists.
    arrow_alive = []

    for j, a in enumerate(T_uniq):
        T_Alive0 = np.argwhere((hp[:, 0] > 0) & (T[:, 0]==a)).flatten()

        alive = ax.quiver(pos[0, 0, T_Alive0], pos[0, 1, T_Alive0],
                          dpos[0, 0, T_Alive0], dpos[0, 1, T_Alive0],
                          color=cm[a], alpha=.5, scale=30,
                          width=0.015, pivot="mid")
        arrow_alive.append(alive)

    points_dead = toflat(
        [ax.plot([], [], 'x', color=cm[a], alpha=.4, markersize=5.) for a in T_uniq])

    # figure things
    fig.tight_layout()
    # close the plot
    plt.close()

    # define init
    def init():
        for j, a in enumerate(T_uniq):
            T_Alive = np.argwhere((hp[:, 0] > 0) & (T[:, 0]==a)).flatten()
            T_Dead = np.argwhere((hp[:, 0] <= 0) & (T[:, 0]==a)).flatten()

            if len(T_Alive) > 0:
                arrow_alive[j].set_UVC(dpos[0, 0, T_Alive], dpos[0, 1, T_Alive])
            if len(T_Dead) > 0:
                points_dead[j].set_data(pos[0, 0, T_Dead], pos[0, 1, T_Dead])
        return ((*arrow_alive, *points_dead))

    # define animate
    def animate(i):
        for j, a in enumerate(T_uniq):
            # get IDs of alive and dead units.
            T_Alive = np.argwhere((hp[:, i] > 0) & (T[:, i]==a)).flatten()
            T_Dead = np.argwhere((hp[:, i] <= 0) & (T[:, i]==a)).flatten()

            if len(T_Alive) > 0:
                # we #'set the offsets' i.e arrow position.
                arrow_alive[j].set_offsets(pos[i, :, T_Alive])
                # we set the directions of the arrows
                arrow_alive[j].set_UVC(dpos[i, 0, T_Alive], dpos[i, 1, T_Alive])
            if len(T_Dead) > 0:
                points_dead[j].set_data(pos[i, 0, T_Dead], pos[i, 1, T_Dead])

        return ((*arrow_alive, *points_dead))

    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=100, frames=N_f, blit=False)
    plt.show()
    return ani


################################# EXTRACT FUNCTIONS ###################################


def extract_pos(results, n):
    return np.stack(([np.array([results[i][j].pos for i in range(len(results))]) for j in range(n)]), axis=2)


def extract_dpos(results, n):
    return np.stack(([np.array([results[i][j].dd for i in range(len(results))]) for j in range(n)]), axis=2)


def extract_hp(results, n):
    return np.vstack(([np.array([results[i][j].hp for i in range(len(results))]) for j in range(n)]))


def extract_team(results, n):
    return np.vstack(([np.array([results[i][j].team for i in range(len(results))]) for j in range(n)]))


########################## MATH FUNCTIONS ##############################################

def euclidean(u_i, u_j):
    return np.sqrt(np.sum(np.square(u_i - u_j)))


def magnitude(p):
    return np.sqrt(np.dot(p, p))


def deriv_norm(u_i, u_j):
    return (u_j - u_i) / euclidean(u_i, u_j)


def dudt(u_i, u_j, s_i):
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


def find_nearest_enemy(u, units):
    # get the alive units, and calculate the euclidean distance from them.
    alive_enemies = [i for i in units if (i.hp > 0) and (i.team != u.team)]
    if len(alive_enemies) > 0:
        D = np.asarray([euclidean(u.pos, i.pos) for i in alive_enemies])
        return alive_enemies[np.argmin(D)]
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


####################### GROUP FUNCTIONS ########################################


def create_unit_group(df, u_type, n, pos_params):
    """
    Given a unit type, database, n sample and position params, create a unit group blob
    using gaussian.

    pos_params should be a tuple (2,) for mean and var.

    we cannot however assign enemies without a list to draw from, so this might have
    to be assigned after this function.
    """

    units = [Unit1(df, u_type) for i in range(n)]
    for u in units:
        u.pos = np.random.normal(*pos_params, size=(2,))
    return units


def assign_enemies(units):
    # get team ids
    for u in units:
        u.target = find_random_enemy(u, units)


######################## MISC #######################################


def time_n(method, r=5, *args, **kws):
    """
    r for repeats
    """
    T = np.zeros(r+1,)
    for i in range(r+1):
        ts = time.time()
        _ = method(*args, **kws)
        T[i] = time.time() - ts
    # drop the first example as it is always slowest
    return T[1:]


def toflat(L):
    """
    Given nd-list, flatten to single dimension.
    """
    return list(it.chain.from_iterable(L))


def colorwheel():
    return ["red", "blue", "green", "orange", "purple", "brown", "black",
            "cyan", "yellow"]