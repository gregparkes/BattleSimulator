#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:30:45 2019

@author: gparkes
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import itertools as it

from .simulator_fast import frame_columns
from .utils import check_columns, slice_loop

from matplotlib.lines import Line2D

# all functions to import
__all__ = ["quiver_fight"]


def _loop_colors():
    return ["red", "blue", "green", "orange", "purple", "brown", "black",
            "cyan", "yellow"]


def quiver_fight(Frames,
                 terrain=None,
                 allegiance_label={},
                 allegiance_color={},
                 quant_size_map={}):
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
    Frames : pd.DataFrame
        The dataframe with each frame step to animate
        Columns included must be: 'x', 'y', 'dir_x', 'dir_y', 'allegiance', 'frame' and 'alive'
    terrain : bsm.Terrain object
        A terrain object to generate and draw from.
    allegiance_label : dict
        maps allegiance in Frames["allegiance"] (k) to a label str (v)
    allegiance_color : dict
        maps allegiance in Frames["allegiance"] (k) to a color str (v)
    quantify_size : dict
        If True, use unit_quant to estimate unit value then set this size to quivers.

    Returns
    ------
    anim : matplotlib.pyplot.animation
        object to animate then from.
    """
    check_columns(Frames, frame_columns())

    # set plt.context
    plt.rcParams["animation.html"] = "html5"
    # dataframe
    N_frames = Frames.index.unique().shape[0]
    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    xmin, xmax, ymin, ymax = terrain.bounds_
    terrain.plot(ax, alpha=.2)

    # find bounds
    ax.set_xlim(xmin-.5, xmax+.5)
    ax.set_ylim(ymin-.5, ymax+.5)

    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # use the numerical allegiance.
    allegiances = Frames["allegiance"].unique()
    # create defaults if the dictionary size does not match the allegiance flags
    if len(allegiance_label) != allegiances.shape[0]:
        allegiance_label = dict(zip(allegiances.tolist(),
                                    ["team%d" % i for i in it.islice(it.count(1), 0, allegiances.shape[0])]))
    if len(allegiance_color) != allegiances.shape[0]:
        allegiance_color = dict(zip(allegiances.tolist(),
                                    slice_loop(_loop_colors(), allegiances.shape[0])))
    # unique units.
    Uunits = Frames["army"].unique()
    if len(quant_size_map) == 0:
        quant_size_map = {k: 1 for k in Uunits}

    combs = list(it.product(allegiances, Uunits))

    """
    Create two groups for each allegiance:
        1. The units that are alive, are arrows.
        2. The units that are dead, are crosses 'x'
    """

    qalive = []
    dead = []

    for a, un in combs:
        f1 = Frames.loc[0].query("(allegiance==@a) & (army==@un) & alive")
        team_alive = ax.quiver(f1.x, f1.y, f1.dir_x, f1.dir_y, color=allegiance_color[a], alpha=.5,
                               scale=30*(quant_size_map[un]+1.), width=0.015, pivot="mid")
        qalive.append(team_alive)

        team_dead, = ax.plot([], [], 'x', color=allegiance_color[a], alpha=.2, markersize=5.)
        dead.append(team_dead)

    # configure legend, extras.
    # design custom legend
    custom_lines = [Line2D([0], [0], color=allegiance_color[a], lw=4) for a in allegiances]
    ax.legend(custom_lines, [allegiance_label[a] for a in allegiances], loc="upper right")
    fig.tight_layout()
    plt.close()


    # an initialisation function = to plot at the beginning.
    def init():
        for j, (a, un) in enumerate(combs):
            new_alive = Frames.loc[0].query("(allegiance==@a) & (army==@un) & alive")
            if len(new_alive) > 0:
                qalive[j].set_UVC(new_alive["dir_x"], new_alive["dir_y"])

        return (*qalive, *dead)


    # animating the graph with step i
    def animate(i):
        for j, (a, un) in enumerate(combs):
            new_alive = Frames.loc[i].query("(allegiance == @a) & (alive) & (army == @un)")
            new_dead = Frames.loc[i].query("(allegiance == @a) & (not alive) & (army == @un)")
            if len(new_alive) > 0:
                qalive[j].set_offsets(np.vstack((new_alive["x"], new_alive["y"])).T)
                qalive[j].set_UVC(new_alive["dir_x"], new_alive["dir_y"])
            if len(new_dead) > 0:
                dead[j].set_data(new_dead["x"], new_dead["y"])

        return (*qalive, *dead)


    return animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=100, frames=N_frames, blit=True)
