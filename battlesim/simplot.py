#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:30:45 2019

@author: gparkes
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# all functions to import
__all__ = ["quiver_fight"]


def quiver_fight(xpd):
    """
    Assumes we have the 'complex' xpd object with 'dir_x' and 'dir_y' columns

    Parameters
    -------
    xpd : pd.DataFrame
        The dataframe with each frame step to animate
        Columns included must be: 'x', 'y', 'dir_x', 'dir_y', 'allegiance', 'frame' and 'alive'

    Returns
    ------
    anim : matplotlib.pyplot.animation
        object to animate then from.
    """
    # set plt.context
    plt.rcParams["animation.html"] = "html5"
    # dataframe
    N_frames = xpd.frame.unique().shape[0]
    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    # find bounds
    ax.set_xlim(xpd.x.min() - 1., xpd.x.max() + 1.)
    ax.set_ylim(xpd.y.min() - 1., xpd.y.max() + 1.)
    # label
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # first frame
    frame0 = xpd.query("(frame==0) & alive")
    # republic points
    rep_alive = ax.quiver(frame0.query("allegiance==0").x,
                          frame0.query("allegiance==0").y,
                          frame0.query("allegiance==0").dir_x,
                          frame0.query("allegiance==0").dir_y,
                          color="red", label="Republic", alpha=.5, scale=30, width=0.015, pivot="mid")
    cis_alive = ax.quiver(frame0.query("allegiance==1").x,
                          frame0.query("allegiance==1").y,
                          frame0.query("allegiance==1").dir_x,
                          frame0.query("allegiance==1").dir_y,
                          color="blue", label="CIS", alpha=.5, scale=30, width=0.015, pivot="mid")
    rep_dead, = ax.plot([], [], "rx", alpha=.2, markersize=5.)
    cis_dead, = ax.plot([], [], "bx", alpha=.2, markersize=5.)
    ax.legend(loc="right")
    fig.tight_layout()
    plt.close()

    def init():
        alive_rep = xpd.query("(frame == 0) & (allegiance == 0) & (alive)")
        alive_cis = xpd.query("(frame == 0) & (allegiance == 1) & (alive)")
        dead_rep = xpd.query("(frame == 0) & (allegiance == 0) & (not alive)")
        dead_cis = xpd.query("(frame == 0) & (allegiance == 1) & (not alive)")
        if len(alive_rep) > 0:
            rep_alive.set_UVC(alive_rep.dir_x, alive_rep.dir_y)
        if len(alive_cis) > 0:
            cis_alive.set_UVC(alive_cis.dir_x, alive_cis.dir_x)
        if len(dead_rep) > 0:
            rep_dead.set_data(dead_rep.x, dead_rep.y)
        if len(dead_cis) > 0:
            cis_dead.set_data(dead_cis.x, dead_cis.y)
        return (rep_alive, cis_alive, rep_dead, cis_dead)

    def animate(i):
        alive_rep = xpd.query("(frame == @i) & (allegiance == 0) & (alive)")
        alive_cis = xpd.query("(frame == @i) & (allegiance == 1) & (alive)")
        dead_rep = xpd.query("(frame == @i) & (allegiance == 0) & (not alive)")
        dead_cis = xpd.query("(frame == @i) & (allegiance == 1) & (not alive)")

        if len(alive_rep) > 0:
            rep_alive.set_offsets(np.vstack((alive_rep.x, alive_rep.y)).T)
            rep_alive.set_UVC(alive_rep.dir_x, alive_rep.dir_y)
        if len(alive_cis) > 0:
            cis_alive.set_offsets(np.vstack((alive_cis.x, alive_cis.y)).T)
            cis_alive.set_UVC(alive_cis.dir_x, alive_cis.dir_y)
        if len(dead_rep) > 0:
            rep_dead.set_data(dead_rep.x, dead_rep.y)
        if len(dead_cis) > 0:
            cis_dead.set_data(dead_cis.x, dead_cis.y)
        return (rep_alive, cis_alive, rep_dead, cis_dead)

    return animation.FuncAnimation(fig, animate, init_func=init,
                                   interval=100, frames=N_frames, blit=True)
