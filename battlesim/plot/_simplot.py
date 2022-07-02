#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:30:45 2019

@author: gparkes
"""
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import itertools as it

from battlesim._utils import slice_loop

from matplotlib.lines import Line2D

# all functions to import
__all__ = ["quiver_fight"]


def quiver_fight(frames: np.ndarray,
                 terrain=None,
                 allegiance_label={},
                 allegiance_color={}):
    """
    Generates an animated quiver plot with units moving around the arena
    and attacking each other. Requires the Frames object as output from a 'battle.simulate()'
    call.

    Units that are alive appear as directional quivers, units that are dead
    appear as crosses 'x'.

    We recommend you use this in conjunction with Jupyter notebook:
        HTML(bsm.quiver_fight(Frames).tojshtml())

    Parameters
    -------
    frames : pd.DataFrame
        The dataframe with each frame step to animate
        Columns included must be: 'x', 'y', 'dir_x', 'dir_y', 'allegiance', 'frame' and 'alive'
    terrain : bsm.Terrain object, optional
        A terra object to generate and draw from.
    allegiance_label : dict
        maps allegiance in Frames["allegiance"] (k) to a label str (v)
    allegiance_color : dict
        maps allegiance in Frames["allegiance"] (k) to a color str (v)

    Returns
    ------
    anim : matplotlib.pyplot.animation
        object to animate then from.
    """
    # set plt.context
    plt.rcParams["animation.html"] = "jshtml"
    # plt.rcParams['animation.writer'] = 'pillow'
    # dataframe
    N_frames = frames.shape[0]
    # create plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # plot the terra underneath
    if terrain is not None:
        # plots the terrain using the object.
        terrain.plot(ax, alpha=.2)

    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # use the numerical allegiance.
    allegiances = np.unique(frames["team"])
    n_allegiances = allegiances.shape[0]

    # create defaults if the dictionary size does not match the allegiance flags
    if len(allegiance_label) != n_allegiances:
        allegiance_label = dict(zip(allegiances.tolist(),
                                    ["team%d" % i for i in it.islice(it.count(1), 0, n_allegiances)]))
    if len(allegiance_color) != n_allegiances:
        allegiance_color = dict(zip(allegiances.tolist(),
                                    slice_loop(colors.BASE_COLORS.keys(), n_allegiances)))
    # unique units.
    Uunits = np.unique(frames['utype'])
    combs = list(it.product(allegiances, Uunits))

    """
    Create two groups for each allegiance:
        1. The units that are alive, are arrows.
        2. The units that are dead, are crosses 'x'
    """

    qalive = []
    dead = []

    for a, un in combs:
        f1 = frames[0][reduce(np.logical_and, [frames[0]['hp'] > 0., frames[0]['utype'] == un, frames[0]['team'] == a])]
        #f1 = frames.loc[0].query("(allegiance==@a) & (army==@un) & alive")
        team_alive = ax.quiver(f1['x'], f1['y'], f1['ddx'], f1['ddy'], color=allegiance_color[a], alpha=.5,
                               scale=30, width=0.015, pivot="mid")
        qalive.append(team_alive)

        team_dead, = ax.plot([], [], 'x', color=allegiance_color[a], alpha=.2, markersize=5.)
        dead.append(team_dead)

    # configure legend, extras.
    # set plot bounds
    xmin, xmax, ymin, ymax = terrain.bounds_
    ax.set_xlim(xmin - .5, xmax + .5)
    ax.set_ylim(ymin - .5, ymax + .5)
    # design custom legend
    custom_lines = [Line2D([0], [0], color=allegiance_color[a], lw=4) for a in allegiances]
    ax.legend(custom_lines, [allegiance_label[a] for a in allegiances], loc="upper right")
    fig.tight_layout()
    plt.close()

    # an initialisation function = to plot at the beginning.
    def _init():
        for j, (_a, _un) in enumerate(combs):
            # replaced query with loc as it's way faster.
            new_alive = frames[0][reduce(np.logical_and, [frames[0]['hp'] > 0., frames[0]['utype'] == _un, frames[0]['team'] == _a])]
            #new_alive = frames.loc[0].query("(allegiance==@_a) & (army==@_un) & alive")
            if new_alive.shape[0] > 0:
                qalive[j].set_UVC(new_alive["ddx"], new_alive["ddy"])

        return (*qalive, *dead)

    # animating the graph with step i
    def _animate(i):
        # i is the frame, aligns with frames.
        for j, (_a, _un) in enumerate(combs):
            # replaced query with loc as it's way faster.
            alive_i = frames[i]['hp'] > 0.
            team_type_i = np.logical_and(frames[i]['team'] == _a, frames[i]['utype'] == _un)

            new_alive = frames[i][np.logical_and(team_type_i, alive_i)]
            new_dead = frames[i][np.logical_and(team_type_i, ~alive_i)]
            #new_alive = frames.loc[i].query("(allegiance == @_a) & (alive) & (army == @_un)")
            #new_dead = frames.loc[i].query("(allegiance == @_a) & (not alive) & (army == @_un)")
            if len(new_alive) > 0:
                qalive[j].set_offsets(np.vstack((new_alive["x"], new_alive["y"])).T)
                # force N to be number of alive samples to prevent error
                qalive[j].N = new_alive.shape[0]
                qalive[j].set_UVC(new_alive["ddx"], new_alive["ddy"])
            if len(new_dead) > 0:
                dead[j].set_data(new_dead["x"], new_dead["y"])

        return (*qalive, *dead)

    return animation.FuncAnimation(fig, _animate, init_func=_init,
                                   interval=100, frames=N_frames, blit=True)
