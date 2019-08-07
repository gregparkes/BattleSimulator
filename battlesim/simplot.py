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

from .simulator import frame_columns
from .utils import check_columns

# all functions to import
__all__ = ["quiver_fight"]


def _loop_colors():
    return ["red", "blue", "green", "orange", "purple", "brown", "black",
            "cyan", "yellow"]


def quiver_fight(Frames, allegiance_label={}, allegiance_color={}):
    """
    Generates an animated quiver plot with units moving around the arena
    and attacking each other. Requires the Frames object as output from a 'battle.simulate()'
    call.

    We recommend you use this in conjunction with Jupyter notebook:
        HTML(bsm.quiver_fight(Frames).tojshtml())

    Parameters
    -------
    Frames : pd.DataFrame
        The dataframe with each frame step to animate
        Columns included must be: 'x', 'y', 'dir_x', 'dir_y', 'allegiance', 'frame' and 'alive'
    allegiance_label : dict
        maps allegiance in Frames["allegiance"] (k) to a label str (v)
    allegiance_color : dict
        maps allegiance in Frames["allegiance"] (k) to a color str (v)

    Returns
    ------
    anim : matplotlib.pyplot.animation
        object to animate then from.
    """
    check_columns(Frames, frame_columns())

    # set plt.context
    plt.rcParams["animation.html"] = "html5"
    # dataframe
    N_frames = Frames["frame"].unique().shape[0]
    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    # find bounds
    ax.set_xlim(Frames["x"].min() - 1., Frames["x"].max() + 1.)
    ax.set_ylim(Frames["y"].min() - 1., Frames["y"].max() + 1.)
    # hide axes labels
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # first frame
    frame0 = Frames.query("(frame==0) & alive")
    # use the numerical allegiance.
    allegiances = Frames["allegiance"].unique()

    # create defaults if the dictionary size does not match the allegiance flags
    if len(allegiance_label) != allegiances.shape[0]:
        allegiance_label = dict(zip(allegiances.tolist(),
                                    ["team%d" % i for i in it.islice(it.count(1), 0, allegiances.shape[0])]))
    if len(allegiance_color) != allegiances.shape[0]:
        allegiance_color = dict(zip(allegiances.tolist(),
                                    list(it.islice(it.cycle(_loop_colors()), 0, allegiances.shape[0]))))
    """
    Create two groups for each allegiance:
        1. The units that are alive, are arrows.
        2. The units that are dead, are crosses 'x'
    """
    alive = []
    dead = []

    # for each allegiance, assign ax quiver, plot object to lists
    for k in allegiances:
        team_alive = ax.quiver(frame0.query("allegiance==@k").x,
                          frame0.query("allegiance==@k").y,
                          frame0.query("allegiance==@k").dir_x,
                          frame0.query("allegiance==@k").dir_y,
                          color=allegiance_color[k],
                          label=allegiance_label[k],
                          alpha=.5, scale=30,
                          width=0.015, pivot="mid")
        alive.append(team_alive)

        team_dead, = ax.plot([], [], 'x', color=allegiance_color[k], alpha=.2, markersize=5.)
        dead.append(team_dead)

    # configure legend, extras.
    ax.legend(loc="right")
    fig.tight_layout()
    plt.close()

    # an initialisation function = to plot at the beginning.
    def init():
        for j, k in enumerate(allegiances):
            new_alive = Frames.query("(frame == 0) & (allegiance == @k) & (alive)")
            new_dead = Frames.query("(frame == 0) & (allegiance == @k) & (not alive)")

            if len(new_alive) > 0:
                alive[j].set_UVC(new_alive["dir_x"], new_alive["dir_y"])
            if len(new_dead) > 0:
                dead[j].set_data(new_dead["x"], new_dead["y"])
        return (*alive, *dead)

    # animating the graph with step i
    def animate(i):
        for j, k in enumerate(allegiances):
            new_alive = Frames.query("(frame == @i) & (allegiance == @k) & (alive)")
            new_dead = Frames.query("(frame == @i) & (allegiance == @k) & (not alive)")

            if len(new_alive) > 0:
                alive[j].set_offsets(np.vstack((new_alive["x"], new_alive["y"])).T)
                alive[j].set_UVC(new_alive["dir_x"], new_alive["dir_y"])
            if len(new_dead) > 0:
                dead[j].set_data(new_dead["x"], new_dead["y"])
        return (*alive, *dead)


    return animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=100, frames=N_frames, blit=True)
