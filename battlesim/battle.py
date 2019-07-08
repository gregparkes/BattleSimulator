#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:39:42 2019

@author: gparkes
"""

import numpy as np
import itertools as it
from scipy import stats

from . import utils
from . import simulator
from .distributions import Distribution


class Battle(object):
    """
    This 'Battle' object provides the interface for the user of simulating
    a number of Battles.

    Each simulation follows a:
        Load -> Create -> Simulate -> Draw
    flow.
    """

    def __init__(self, dbfilepath):
        """
        Instantiate this object with a filepath leading to
        """
        assert isinstance(dbfilepath, str), "dbfilepath must be of type ('str')"
        self.db_ = utils.import_and_check_unit_file(dbfilepath)
        self.M_ = None


    """---------------------- FUNCTIONS --------------------"""

    def create_army(self, army_set):
        """
        Armies are groupings of (<'Unit Type'>, <number of units>). You can
        create one or more of these.

        We make use of the dataset (`db`) with army_set.

        We create the 'M' matrix, which is directly fed into any 'simulation' function.

        Parameters
        -------
        army_set : list of 2-tuple
            A list of 'army groups' given as ('Unit Type', number of units)

        Returns self
        -------
        self
        """
        #assign
        assert isinstance(army_set, (list, tuple)), "army_set must be of type ('list','tuple')"
        self.army_set_ = army_set
        self.N_ = sum([arm[1] for arm in army_set])

        self.M_ = np.zeros((self.N_), dtype=[
            ("team",int,1),("pos",float,2),("hp",float,1),
            ("range",float,1),("speed",float,1),("acc",float,1),
            ("dodge",float,1),("dmg",float,1),("target",int,1)
        ])

        segments = utils.get_segments(army_set)
        allg = np.asarray([self.db_.loc[U[0],"allegiance_int"] for U in army_set])

        # set initial values.
        for (u, n), (start, end), team in zip(army_set, segments, allg):
            self.M_["team"][start:end] = team
            self.M_["hp"][start:end] = self.db_.loc[u,"HP"]
            self.M_["range"][start:end] = self.db_.loc[u,"Range"]
            self.M_["speed"][start:end] = self.db_.loc[u,"Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u,"Miss"]/100.
            self.M_["acc"][start:end] = self.db_.loc[u,"Accuracy"]/100.
            self.M_["dmg"][start:end] = self.db_.loc[u,"Damage"]
            # random target - first we filter out segments in our team, then unpack xegment tuple into numpy.arange, then cat together indices.
            targets = np.hstack([np.arange(*(s[1]))
                for s in it.filterfalse(lambda x: x[0]==team, zip(allg, segments))])
            self.M_["target"][start:end] = np.random.choice(targets, size=(n,))

        return self


    def position_from_dist(self, distributions):
        """
        Assign locations to each 'army set' using a distribution from the battlesim.Distribution object.

        Parameters
        -------
        distributions : battlesim.Distribution or list of battlesim.Distribution.
            The distribution(s) corresponding to each group.

        Returns
        -------
        self
        """
        segments = utils.get_segments(self.army_set_)
        if isinstance(distributions, Distribution):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                self.M_["pos"][start:end] = distributions.sample(n)
        else:
            for (u, n), (start, end), d in zip(self.army_set_, segments, distributions):
                self.M_["pos"][start:end] = d.sample(n)
        return self


    @utils.deprecated
    def position_gaussian(self, pos_set):
        """
        Assign positions given as (mean, var) for gaussian locations for each
        army set.

        Returns self
        """
        segments = utils.get_segments(self.army_set_)
        for (u, n), (mean, var), (start, end) in zip(self.army_set_, pos_set, segments):
            self.M_["pos"][start:end] = np.random.normal(mean, var, size=(n,2))
        return self


    def simulate(self, **kwargs):
        """
        Runs the 'simulate_battle' algorithm. Creates and passes a copy to simulate..

        Returns pd.DataFrame of frames.
        """
        return simulator.simulate_battle(np.copy(self.M_), **kwargs)

    """ ---------------------- MISC ------------------------------ """

