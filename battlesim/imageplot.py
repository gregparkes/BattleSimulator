#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:38:51 2019

@author: gparkes
"""
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["accuracy_contour", "hit_contour"]

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
    xlim = np.array([unit.x_ - 15, unit.x_ + 15])
    ylim = np.array([unit.y_ - 15, unit.y_ + 15])

    xgrid = np.linspace(xlim[0], xlim[1], 100)
    ygrid = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    Z = np.sqrt(np.power(X - unit.x_, 2) + np.power(Y - unit.y_, 2))
    # normalize
    n_Z = (1. - (Z / acc_penalty)) * unit.accuracy_
    # get range also

    fig,ax=plt.subplots()
    ax.set_title("Normalized hit chance drop-off for '{}', ignoring enemy dodge.".format(unit.unit_type_))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(unit.x_, unit.y_, c="red")
    CS = ax.contour(X, Y, n_Z, levels=15)
    # add range circle
    ax.add_patch(plt.Circle((unit.x_, unit.y_), unit.range_, color="green", alpha=.3))
    ax.clabel(CS, inline=1, fontsize=8)
    return fig, ax


def hit_contour(u1, u2, acc_pen=15., boxlim=10):
    """
    Given two units, simulate the contours for hitting each other.
    """
    def contour_grid(u):
        xgrid = np.linspace(u.x_ - boxlim, u.x_ + boxlim, 100)
        ygrid = np.linspace(u.y_ - boxlim, u.y_ + boxlim, 100)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = np.sqrt(np.power(X - u.x_, 2) + np.power(Y - u.y_, 2))
        return X, Y, Z

    u1x, u1y, u1z = contour_grid(u1)
    u2x, u2y, u2z = contour_grid(u2)
    # normalize
    nZu1 = (1. - (u1z / acc_pen)) * u1.accuracy_ * (1. - u2.dodge_)
    nZu2 = (1. - (u2z / acc_pen)) * u2.accuracy_ * (1. - u1.dodge_)

    fig,ax=plt.subplots()
    # scatter
    ax.scatter(u1.x_, u1.y_, c="red", s=50)
    ax.scatter(u2.x_, u2.y_, c="green", marker="^", s=50)
    # add range circle
    ax.add_patch(plt.Circle((u1.x_, u1.y_), u1.range_, color="green", alpha=.3))
    ax.add_patch(plt.Circle((u2.x_, u2.y_), u2.range_, color="red", alpha=.3))
    # contours
    CS1 = ax.contour(u1x, u1y, nZu1, levels=10, cmap="winter")
    CS2 = ax.contour(u2x, u2y, nZu2, levels=10, cmap="summer")
    ax.clabel(CS1, inline=1, fontsize=6)
    ax.clabel(CS2, inline=1, fontsize=6)
    return fig, ax
