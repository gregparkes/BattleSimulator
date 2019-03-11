#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:24:48 2019

@author: gparkes
"""

import numpy as np

from . import unit
from . import utils
from . import ai

class Army(object):
    """
    An army object consists of a group of Units, and allows
    for assignment of army positions.
    """
    def __init__(self, b, unit_type, n):
        """
        Parameters
        -------
        b : Battle object
            the battle this army is taking part in
        unit_type : str
            type of unit
        N : int
            Number of units

        Returns
        -------
        self
        """
        self.unit_type_ = unit_type
        assert n > 0, "Parameter 'n' must be > 0"
        self.units_ = [unit.Unit(b, unit_type) for i in range(n)]
        # assign default pos
        self.set_loc_gaussian([0., 0.], [1., 1.])

        return


    def __getitem__(self, index):
        return self.units_[index]

    def _get_army_size(self):
        return len(self.units_)

    def _set_army_size(self, newN):
        self._N = newN

    def _get_army_alive(self):
        return self._get_army_alive_count() > 0

    def _get_army_alive_count(self):
        return sum([u.alive_ for u in self.units_])

    def _get_army_x(self):
        return np.asarray([u.pos_[0] for u in self.units_])

    def _get_army_y(self):
        return np.asarray([u.pos_[1] for u in self.units_])

    def _get_army_positions(self):
        return np.vstack(([self.units_[i].pos_ for i in range(self.N_)]))

    ############## PROPERTIES ####################################

    N_ = property(_get_army_size, _set_army_size, doc="The size of the army")
    pos_ = property(_get_army_positions)
    x_ = property(_get_army_x)
    y_ = property(_get_army_y)
    alive_ = property(_get_army_alive)
    remaining_ = property(_get_army_alive_count)


    def set_loc_gaussian(self, mean, sd):
        """
        Allocate Unit positions based on a gaussian distribution (centred around
        the mean).

        Parameters
        -------
        mean : np.ndarray (2,)
            the mean for x [0] and y[1] position
        sd : np.ndarray (2,)
            the standard deviation for x[0] and y[1] position

        Returns
        -------
        self
        """
        pos = np.random.normal(loc=mean, scale=sd, size=(self.N_, 2))
        # assign to each unit
        for i,u in enumerate(self.units_):
            u.set_position(pos[i,:])
        return self


    def set_loc_grid(self, xlim, ylim, theta=0.):
        """
        Allocate Unit positions based on a uniform grid (with an optional rotation).

        Parameters
        -------
        xlim : tuple, list (2,)
            The lower and upper bounds of the x axis
        ylim : tuple, list (2,)
            The lower and upper bounds of the y axis
        theta : float
            The amount of rotation to apply to the grid (optional)

        Returns
        -------
        self
        """
        # 2-d rotation matrix
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        # find two numbers we like
        factors = np.sort(utils.factor(self.N_))
        nx = factors[len(factors)//2]
        ny = self.N_ // nx

        X, Y = utils.grid_of_points(xlim, ylim, nx, ny)
        # stack X, Y together, then transform with rotation
        R_Z = np.dot(R, np.vstack((X.reshape(self.N_,order="F"),
                                   Y.reshape(self.N_,order="F"))))
        # assign R_Z positions to each unit
        for i,u in enumerate(self.units_):
            u.set_position(R_Z[:,i])
        return self


    def set_ai_target(self, ai_func):
        """
        Choose the AI algorithm to target enemies.

        Parameters
        -------
        ai_func : function
            The function from the .ai package to use, applied to all units.
        """
        for i,u in enumerate(self.units_):
            u.ai_ = ai_func
        return self


    def __repr__(self):
        return "Army(type='%s', n='%d')" % (self.unit_type_, self.N_)



class DelayArmy(object):
    """
    An army object consists of a group of Units, and allows
    for assignment of army positions. This object does not
    create these objects immediately but stores them for the Battle
    object to create.
    """
    def __init__(self, name, N=10):
        """
        Create an army unit of type 'name' with N count.
        """
        self._name = name
        assert N > 0, "Parameter 'N' must be > 0"
        self._N = N
        # defaults
        self._pos_type = "gaussian"
        self._pos_params = {"loc":[0., 0.], "scale":[1., 1.]}
        self._ai_func = "random"
        self._idx_range = None

    def _get_N(self):
        return self._N

    def _get_name(self):
        return self._name

    def _assign_position(self):
        return {self._pos_type: self._pos_params}

    def _get_target_ai(self):
        return self._ai_func

    def _get_index_range(self):
        return self._idx_range

    def _set_index_range(self, idx_range):
        self._idx_range = idx_range

    N_ = property(_get_N)
    name_ = property(_get_name)
    pos_ = property(_assign_position)
    ai_ = property(_get_target_ai)
    index_range_ = property(_get_index_range, _set_index_range)

    ################## FUNCTIONS ####################################

    def set_loc_gaussian(self, mean, sd):
        """
        Allocate Unit positions based on a gaussian distribution (centred around
        the mean).

        Parameters
        -------
        mean : np.ndarray (2,)
            the mean for x [0] and y[1] position
        sd : np.ndarray (2,)
            the standard deviation for x[0] and y[1] position

        Returns
        -------
        self
        """
        self._pos_type = "gaussian"
        self._pos_params = {"loc":list(mean), "scale":list(sd)}
        return self


    def set_loc_grid(self, xlim, ylim, theta=0.):
        """
        Allocate Unit positions based on a uniform grid (with an optional rotation).

        Parameters
        -------
        xlim : tuple, list (2,)
            The lower and upper bounds of the x axis
        ylim : tuple, list (2,)
            The lower and upper bounds of the y axis
        theta : float
            The amount of rotation to apply to the grid (optional)

        Returns
        -------
        self
        """
        self._pos_type = "grid"
        self._pos_params = {"xlim":list(xlim), "ylim":list(ylim), "theta":theta}
        return self


    def set_ai_target(self, ai_func):
        """
        Choose the AI algorithm to target enemies.

        Parameters
        -------
        ai_func : function
            The function from the .ai package to use.
        """
        accepted_ai = ["random", "nearest"]
        if ai_func not in accepted_ai:
            raise ValueError("'{}' not found in accepted ai".format(ai_func))

        self._ai_func = ai_func
        return self


    def set_ai_initial(self, ai_init_func):
        """
        Choose the AI algorithm to find an initial target.

        Parameters
        -------
        ai_init_func : str
            The function from the .ai package to use.
        """

        accepted_ai = ["random", "nearest"]

        if ai_init_func not in accepted_ai:
            raise ValueError("'{}' not found in accepted ai".format(ai_init_func))

        self._ai_init_func = ai_init_func
        return self


    def __repr__(self):
        return "DelayArmy(type='%s', n='%d', pos='%s')" % (self._name, self._N, self._pos_type)

