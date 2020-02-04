#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:38:51 2019

@author: gparkes
"""
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["accuracy_contour", "hit_contour", "quiver_frame"]

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


def accuracy_contour(M, index, acc_penalty=15., boxlim=15):
    """
    Given a selected unit, with it's position, simulate accuracy/miss distances
    to draw accuracy contours.
    """
    xlim = np.asarray([M["pos"][index, 0] - boxlim, M["pos"][index, 0] + boxlim])
    ylim = np.asarray([M["pos"][index, 1] - boxlim, M["pos"][index, 1] + boxlim])

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.sqrt(np.power(X - M["pos"][index, 0], 2) + np.power(Y - M["pos"][index, 1], 2))
    # normalize
    n_Z = (1. - (Z / acc_penalty)) * M["acc"][index]
    # get range also

    fig,ax=plt.subplots()
    ax.set_title("Normalized hit chance drop-off, ignoring enemy dodge.")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(M["pos"][index, 0], M["pos"][index, 1], c="red")
    CS = ax.contour(X, Y, n_Z, levels=15)
    # add range circle
    ax.add_patch(plt.Circle((M["pos"][index, 0], M["pos"][index, 1]), M["range"][index], color="green", alpha=.3))
    ax.clabel(CS, inline=1, fontsize=8)
    return fig, ax


def hit_contour(M, i1, i2, acc_penalty=15., boxlim=10):
    """
    Given two units, simulate the contours for hitting each other.
    """
    def contour_grid(M, j):
        xgrid = np.linspace(M["pos"][j, 0] - boxlim, M["pos"][j, 0] + boxlim, 100)
        ygrid = np.linspace(M["pos"][j, 1] - boxlim, M["pos"][j, 1] + boxlim, 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = np.sqrt(np.power(X - M["pos"][j, 0], 2) + np.power(Y - M["pos"][j, 1], 2))
        return X, Y, Z

    u1x, u1y, u1z = contour_grid(M, i1)
    u2x, u2y, u2z = contour_grid(M, i2)
    # normalize
    nZu1 = (1. - (u1z / acc_penalty)) * M["acc"][i1] * (1. - M["dodge"][i2])
    nZu2 = (1. - (u2z / acc_penalty)) * M["acc"][i2] * (1. - M["dodge"][i1])

    fig,ax=plt.subplots()
    # scatter
    ax.scatter(M["pos"][i1, 0], M["pos"][i1, 1], c="red", s=50)
    ax.scatter(M["pos"][i2, 0], M["pos"][i2, 1], c="green", marker="^", s=50)
    # add range circle
    ax.add_patch(plt.Circle((M["pos"][i1, 0], M["pos"][i1, 1]), M["range"][i1], color="green", alpha=.3))
    ax.add_patch(plt.Circle((M["pos"][i2, 0], M["pos"][i2, 1]), M["range"][i2], color="red", alpha=.3))
    # contours
    CS1 = ax.contour(u1x, u1y, nZu1, levels=10, cmap="winter")
    CS2 = ax.contour(u2x, u2y, nZu2, levels=10, cmap="summer")
    ax.clabel(CS1, inline=1, fontsize=6)
    ax.clabel(CS2, inline=1, fontsize=6)
    return fig, ax
