#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:38:51 2019

@author: gparkes
"""
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["accuracy_contour"]

################### PLOT FUNCTIONS HERE #######################################

def quiver_frame(xpd, frame_i=0):
    """
    Returns a single frame of quiver_fight.
    """
    # create plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.set_xlim(xpd.x.min() - 1., xpd.x.max() + 1.)
    ax.set_ylim(xpd.y.min() - 1., xpd.y.max() + 1.)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    frame0 = xpd.query("(frame==@frame_i) & alive")
    ax.quiver(frame0.query("allegiance==0").x,
                          frame0.query("allegiance==0").y,
                          frame0.query("allegiance==0").dir_x,
                          frame0.query("allegiance==0").dir_y,
                          color="red", label="Republic", alpha=.5,
                          scale=30, width=0.015, pivot="mid")
    ax.quiver(frame0.query("allegiance==1").x,
                          frame0.query("allegiance==1").y,
                          frame0.query("allegiance==1").dir_x,
                          frame0.query("allegiance==1").dir_y,
                          color="blue", label="CIS", alpha=.5, scale=30,
                          width=0.015, pivot="mid")
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()
    return fig


def accuracy_contour(unit, acc_penalty=15.):
    """
    Given a selected unit, with it's position, simulate accuracy/miss distances
    to draw accuracy contours.
    """
    xlim = np.array([unit.x_ - 10, unit.x_ + 10])
    ylim = np.array([unit.y_ - 10, unit.y_ + 10])

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.sqrt(np.power(X - unit.x_, 2) + np.power(Y - unit.y_, 2))
    # normalize
    n_Z = (1. - (Z / acc_penalty)) * unit.accuracy_

    fig,ax=plt.subplots()
    ax.set_title("Normalized hit chance drop-off for this unit, ignoring enemy dodge.")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(unit.x_, unit.y_, c="red")
    CS = ax.contour(X, Y, n_Z)
    ax.clabel(CS, inline=1, fontsize=8)
    return



def barplot_wins(set_of_battles):
    """
    Plot a barplot of the wins/losses of Republic/CIS
    """
    ind = [0, 1]
    xx = pd.DataFrame(set_of_battles)
    plt.bar(ind, (xx.cis.gt(xx.republic).sum(), xx.cis.lt(xx.republic).sum()))
    plt.ylabel("Number of wins")
    plt.xticks(ind, xx.columns)
    plt.show()
    return