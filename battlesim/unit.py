#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:23:43 2019

@author: gparkes
"""
import numpy as np


class Unit(object):
    """
    Here we create an instance of a generic 'Unit' that can be used.

    The Unit represents a character that will move around in 2-d space on the map, and interact
    with other Units.
    """
    def __init__(self, b, utype=None):
        """
        Parameters
        -------
        b : Battle object
            the battle this Unit should be involved in
        utype : str
            The type of unit we are, in relation to db.

        Returns
        -------
        self
        """
        # extract db from b.
        self.db = b.db
        if utype is None:
            # choose first.
            self.unit_type_ = self.db.index[0]
        else:
            self.unit_type_ = utype

        self.set_params(self.unit_type_)
        pass


    def _get_position(self):
        return self._pos

    def _get_x(self):
        return self._pos[0]

    def _get_y(self):
        return self._pos[1]

    def set_position(self, pos):
        """
        Set the position of x and y.
        """
        self._pos[0] = pos[0]
        self._pos[1] = pos[1]
        return self

    def _is_alive(self):
        return self.curr_hp_ > 0

    def _get_target(self):
        return self._target

    def _set_target(self, unit):
        self._target = unit

    def _get_allegiance(self):
        return self.db.loc[self.unit_type_, "Allegiance"]

    def _get_allegiance_int(self):
        return self.db.loc[self.unit_type_, "allegiance_int"]

    def _get_range(self):
        return self.db.loc[self.unit_type_, "Range"]

    def _get_movement_speed(self):
        return self.db.loc[self.unit_type_, "Movement Speed"]

    def _get_hit_chance(self):
        return self.db.loc[self.unit_type_, "Miss"] / 100.

    def _get_accuracy(self):
        return self.db.loc[self.unit_type_, "Accuracy"] / 100.

    def _get_primary_damage(self):
        return self.db.loc[self.unit_type_, "Damage"]

    def _set_position(self, pos):
        if not isinstance(pos, (list, tuple, np.ndarray)):
            raise TypeError("pos cannot be type '{}'".format(type(pos)))
        if len(pos) != 2:
            raise ValueError("pos must be length 2")
        self._pos = np.asarray(pos)

    def _get_ai(self):
        return self._ai

    def _set_ai(self, ai):
        self._ai = ai

    ############## PROPERTIES ################################

    alive_ = property(_is_alive)
    pos_ = property(_get_position, _set_position)
    x_ = property(_get_x)
    y_ = property(_get_y)
    allegiance_ = property(_get_allegiance)
    allegiance_int_ = property(_get_allegiance_int)
    target_ = property(_get_target, _set_target)
    range_ = property(_get_range)
    move_speed_ = property(_get_movement_speed)
    dodge_ = property(_get_hit_chance)
    accuracy_ = property(_get_accuracy)
    damage_ = property(_get_primary_damage)
    ai_ = property(_get_ai, _set_ai)


    def set_params(self, unit_type=None):
        """
        Given parameters:
            unit_type

        Fill these parameters for each unit.
        """
        if unit_type is None:
            # choose first
            unit_type = self.db.index[0]
        assert unit_type in self.db.index, "unit_type {} does not exist".format(unit_type)
        self.unit_type_ = unit_type
        self.curr_hp_ = self.db.loc[unit_type,"HP"]
        self._target = None
        # generate random location
        self._pos = np.array([0.0, 0.0])
        return self


    def __repr__(self):
        return ("Unit('%s')" % (self.unit_type_))
