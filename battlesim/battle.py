#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:39:42 2019

@author: gparkes
"""

import numpy as np
import pandas as pd

from . import utils
from . import simulator
from . import ai
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
        self.n_armies_ = len(army_set)
        self.N_ = sum([arm[1] for arm in army_set])

        self.M_ = np.zeros((self.N_), dtype=[
            ("team",np.uint8,1),("group",np.uint8,1),("pos",np.float64,2),("hp",np.float64,1),
            ("range",np.float64,1),("speed",np.float64,1),("acc",np.float64,1),
            ("dodge",np.float64,1),("dmg",np.float64,1),("target",np.int64,1)
        ])

        segments = utils.get_segments(army_set)
        self.teams_ = np.asarray([self.db_.loc[U[0],"allegiance_int"] for U in army_set])
        self.allegiances_ = (self.db_[["Allegiance","allegiance_int"]]
                                .set_index("allegiance_int")
                                .drop_duplicates().squeeze())


        # set initial values.
        for i, ((u, n), (start, end), team) in enumerate(zip(army_set, segments, self.teams_)):
            self.M_["team"][start:end] = team
            self.M_["group"][start:end] = i
            self.M_["hp"][start:end] = self.db_.loc[u,"HP"]
            self.M_["range"][start:end] = self.db_.loc[u,"Range"]
            self.M_["speed"][start:end] = self.db_.loc[u,"Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u,"Miss"]/100.
            self.M_["acc"][start:end] = self.db_.loc[u,"Accuracy"]/100.
            self.M_["dmg"][start:end] = self.db_.loc[u,"Damage"]
            # default position: uniform
            dist = Distribution('uniform')
            self.M_["pos"][start:end] = dist.sample(n)

        # initialise AIs as random.
        self.set_initial_ai(["random"]*self.n_armies_)
        self.set_rolling_ai(["random"]*self.n_armies_)

        return self


    def apply_position(self, distributions):
        """
        Assign locations to each 'army set' using a distribution from the battlesim.Distribution object.

        Parameters
        -------
        distributions : str, battlesim.Distribution or list of battlesim.Distribution.
            The distribution(s) corresponding to each group.

        Returns
        -------
        self
        """
        segments = utils.get_segments(self.army_set_)
        if isinstance(distributions, str):
            if distributions in Distribution._get_dists():
                # create a distribution object.
                dist = Distribution(distributions)
                for (u, n), (start, end) in zip(self.army_set_, segments):
                    self.M_["pos"][start:end] = dist.sample(n)
            else:
                raise ValueError("distribution '{}' not found in bsm.Distribution.".format(distributions))
        elif isinstance(distributions, Distribution):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                self.M_["pos"][start:end] = distributions.sample(n)
        elif isinstance(distributions, (list, tuple)):
            for (u, n), (start, end), d in zip(self.army_set_, segments, distributions):
                self.M_["pos"][start:end] = d.sample(n)
        return self


    def set_position(self, positions):
        """
        Set locations to each 'army set' using direct positional coordinates in a numpy.array_like object.

        Parameters
        -------
        positions : list of array_like (n,2)
            The x:y coordinates for each army group.

        Returns
        -------
        self
        """
        segments = utils.get_segments(self.army_set_)
        for (u, n), (start, end), location in zip(self.army_set_, segments, positions):
            self.M_["pos"][start:end] = location.copy()
        return self


    def set_ai(self, initial_functions, rolling_functions):
        """
        An aggregate function for setting all of the AI functionalities of each
        'army set'.

        Parameters
        --------
        initial_functions : list of str
            The AI function to choose for initial targets for each army set. Can choose from:
            ['random', 'pack', 'nearest']
        rolling_functions : list of str
            The AI function to choose for targets when their target dies
            for each army set. Can choose from: ['random', 'pack', 'nearest']

        Returns
        -------
        self
        """
        self.set_initial_ai(initial_functions)
        self.set_rolling_ai(rolling_functions)
        return self


    def set_initial_ai(self, func_names):
        """
        Set the AI decisions for choosing a target initially. Pass a function name from bsm.ai
        for each 'army set'.

        Parameters
        --------
        func_names : list of str
            The AI function to choose for initial targets for each army set. Can choose from:
                ['random', 'pack', 'nearest']

        Returns
        -------
        self
        """
        if isinstance(func_names, (list, tuple)) and (len(func_names) == self.n_armies_):
            self.init_ai_ = dict(zip(range(self.n_armies_), func_names))
        else:
            raise ValueError("ai_funcs is wrong type or length.")

        if self.M_ is None:
            raise TypeError("'M' must be initialised.")

        f_dict = ai.get_map_functions()
        segments = utils.get_segments(self.army_set_)
        # create valid_targets set
        valid_targets = [np.argwhere((self.M_["team"]!=T)).flatten() for T in np.unique(self.M_["team"])]
        valid_allies = [np.argwhere((self.M_["team"]==T)).flatten() for T in np.unique(self.M_["team"])]

        for (u, n), (start, end), func, team in zip(self.army_set_, segments, func_names, self.teams_):
            for i in range(start, end):
                self.M_["target"][i] = f_dict[func](valid_targets[team], valid_allies[team], self.M_, i)
        return self


    def set_rolling_ai(self, func_names):
        """
        Set the AI decisions for choosing a target rolling through the simulation.
        Pass a function name from bsm.ai for each 'army set'.

        Parameters
        --------
        func_names : list of str
            The AI function to choose for targets when their target dies
            for each army set. Can choose from: ['random', 'pack', 'nearest']

        Returns
        -------
        self
        """
        self.rolling_ai_ = dict(zip(range(self.n_armies_), func_names))
        # map these strings to actual functions, ready for simulate.
        mappp = ai.get_map_functions()
        self._mapped_ai = dict(zip(range(self.n_armies_), [mappp[self.rolling_ai_[f]] for f in self.rolling_ai_]))
        return self


    @utils.deprecated
    def apply_position_uniform(self, pos_set):
        """
        Assigns positions given as (low, high) for uniform locations for each
        army set.

        Returns self
        """
        segments = utils.get_segments(self.army_set_)
        for (u, n), (low, high), (start, end) in zip(self.army_set_, pos_set, segments):
            self.M_["pos"][start:end] = low + np.random.rand(n, 2)*(high-low)
        return self


    @utils.deprecated
    def apply_position_gaussian(self, pos_set):
        """
        Assign positions given as (mean, var) for gaussian locations for each
        army set.

        Returns self
        """
        segments = utils.get_segments(self.army_set_)
        for (u, n), (mean, var), (start, end) in zip(self.army_set_, pos_set, segments):
            self.M_["pos"][start:end] = np.random.normal(mean, var, size=(n,2))
        return self


    """ ----------------------------- SIMULATION ----------------------------- """

    def simulate(self, **kwargs):
        """
        Runs the 'simulate_battle' algorithm. Creates and passes a copy to simulate..

        Returns pd.DataFrame of frames.
        """
        if self.M_ is None:
            raise ValueError("Army is not initialised, cannot simulate!")
        else:
            return simulator.simulate_battle(np.copy(self.M_), self._mapped_ai, **kwargs)


    def simulate_k(self, k=10, **kwargs):
        """
        Runs the 'simulate_battle' algorithm 'k' times. Creates and passes a copy
        to simulate.

        Parameters
        --------
        k : int
            The number of iterations. Must be at least 1.
        **kwargs : dict
            keyword arguments to pass to simulate_battle.

        Returns
        -------
        x : pd.DataFrame
            The iteration (k), with the team victorious, and number of units remaining

        Returns the victory for each k iteration, for each team.
        """
        if self.M_ is None:
            raise ValueError("Army is not initialised, cannot simulate!")
        else:
            Z = np.zeros((k,2), dtype=np.int64)
            for i in range(k):
                team_counts = simulator.simulate_battle(np.copy(self.M_), self._mapped_ai,
                                                  ret_frames=False, **kwargs)
                Z[i, :] = team_counts
            return pd.DataFrame(Z, columns=self.allegiances_.values)


    """ ---------------------- MISC ------------------------------ """

