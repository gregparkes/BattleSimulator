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
from . import target
from . import simplot
from .distributions import dist, Distribution, get_options


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
        self.sim_ = None
        # convert db_ index to lower case.
        self.db_names_ = self.db_.index.tolist()
        self.db_.index = self.db_.index.str.lower()


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
        assert utils.is_twotuple(army_set, str, int), "army_set must be a 2-tuple input"

        # convert string elements to lowercase.
        self.army_set_ = [(u.lower(), n) for u, n in army_set]
        self.n_armies_ = len(self.army_set_)
        self.N_ = sum([arm[1] for arm in self.army_set_])

        # check that groups exist in army_set
        utils.check_groups_in_db(self.army_set_, self.db_)

        self.M_ = np.zeros((self.N_), dtype=[
            ("team",np.uint8,1),("group",np.uint8,1),("pos",np.float64,2),("hp",np.float64,1),
            ("range",np.float64,1),("speed",np.float64,1),("acc",np.float64,1),
            ("dodge",np.float64,1),("dmg",np.float64,1),("target",np.int64,1)
        ])

        segments = utils.get_segments(army_set)
        self.teams_ = np.asarray([self.db_.loc[U[0],"allegiance_int"] for U in self.army_set_])
        self.allegiances_ = (self.db_[["Allegiance","allegiance_int"]]
                                .set_index("allegiance_int")
                                .drop_duplicates().squeeze())


        # set initial values.
        for i, ((u, n), (start, end), team) in enumerate(zip(self.army_set_, segments, self.teams_)):
            self.M_["team"][start:end] = team
            self.M_["group"][start:end] = i
            self.M_["hp"][start:end] = self.db_.loc[u,"HP"]
            self.M_["range"][start:end] = self.db_.loc[u,"Range"]
            self.M_["speed"][start:end] = self.db_.loc[u,"Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u,"Miss"]/100.
            self.M_["acc"][start:end] = self.db_.loc[u,"Accuracy"]/100.
            self.M_["dmg"][start:end] = self.db_.loc[u,"Damage"]
            # default position: uniform
            D = Distribution('uniform')
            self.M_["pos"][start:end] = D.sample(n)

        # initialise AIs as random.
        self.set_initial_ai(["random"]*self.n_armies_)
        self.set_rolling_ai(["random"]*self.n_armies_)

        return self


    def apply_position(self, distributions):
        """
        Assign locations to each 'army set' using a distribution from the battlesim.Distribution object.

        e.g

        battle.apply_distribution("gaussian")
        battle.apply_distribution(bsm.Distribution("normal",loc=0,scale=1))
        battle.apply_distribution([bsm.Distribution("beta"), bsm.Distribution("normal")])
        battle.apply_distribution([
            {"name":"beta", "x_loc":0., "y_loc":1.},
            {"dist":"normal", "x_loc":10., "y_loc": 10.}
        ])

        Parameters
        -------
        distributions : str, battlesim.Distribution or list/tuple of battlesim.Distribution/dict.
            The distribution(s) corresponding to each group.
            e.g
            str : distribution name for all army sets
            dict : distribution name and parameters for all army sets
            Distribution : object for all army sets
            list/tuple:
                Distribution : object for each army set
                dict : name and parameters for each army set
                str : dist name for each army set

        Returns
        -------
        self
        """
        # if none for distributions, apply same as before.
        if self.M_ is None:
            raise AttributeError("create_army() has not been called, no positions to allocate")

        segments = utils.get_segments(self.army_set_)
        if isinstance(distributions, str):
            if distributions in get_options()["names"]:
                # create a distribution object.
                D = Distribution(distributions)
                for (u, n), (start, end) in zip(self.army_set_, segments):
                    self.M_["pos"][start:end] = D.sample(n)
            else:
                raise ValueError("distribution '{}' not found in bsm.Distribution.".format(distributions))
        elif isinstance(distributions, Distribution):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                self.M_["pos"][start:end] = distributions.sample(n)
        elif isinstance(distributions, dict):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                D = dist(**distributions)
                self.M_["pos"][start:end] = D.sample(n)
        elif isinstance(distributions, (list, tuple)):
            for (u, n), (start, end), d in zip(self.army_set_, segments, distributions):
                if isinstance(d, Distribution):
                    self.M_["pos"][start:end] = d.sample(n)
                elif isinstance(d, dict):
                    # unpack keywords into the 'dist' function of Distribution
                    D = dist(**d)
                    self.M_["pos"][start:end] = D.sample(n)
                elif isinstance(d, str):
                    # each is a string distribution
                    if d in get_options()["names"]:
                        # create a distribution object.
                        D = Distribution(d)
                        self.M_["pos"][start:end] = D.sample(n)
                    else:
                        raise ValueError("distribution '{}' not found in bsm.Distribution.".format(d))
                else:
                    raise TypeError("Each element of 'distributions' must be a bsm.Distribution or dict")
        else:
            raise TypeError("distributions must be of type [str, Distribution, list, tuple]")
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
        if self.M_ is None:
            raise AttributeError("create_army() has not been called, no positions to allocate to")
        if not isinstance(positions, np.ndarray):
            raise TypeError("positions must be of type [np.ndarray]")
        if positions.shape[1] != 2:
            raise ValueError("size of position dimensions must be 2, not {}".format(positions.shape[1]))
        # check that size meets army_set
        if positions.shape[0] != self.M_.shape[0]:
            raise ValueError("size of position samples must be {}, not {}".format(self.M_.shape[0], positions.shape[0]))

        segments = utils.get_segments(self.army_set_)
        for (u, n), (start, end), location in zip(self.army_set_, segments, positions):
            self.M_["pos"][start:end] = location.copy()
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
            raise AttributeError("ai_funcs is wrong type or length.")

        if self.M_ is None:
            raise TypeError("'M' must be initialised.")
        utils.check_in_list(target.get_init_function_names(), func_names)
        utils.check_list_type(func_names, str)

        f_dict = target.get_map_functions()
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
        if isinstance(func_names, (list, tuple)) and (len(func_names) == self.n_armies_):
            self.init_ai_ = dict(zip(range(self.n_armies_), func_names))
        else:
            raise AttributeError("ai_funcs is wrong type or length.")

        if self.M_ is None:
            raise TypeError("'M' must be initialised.")
        utils.check_in_list(target.get_init_function_names(), func_names)
        utils.check_list_type(func_names, str)

        self.rolling_ai_ = dict(zip(range(self.n_armies_), func_names))
        # map these strings to actual functions, ready for simulate.
        mappp = target.get_map_functions()
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
            raise AttributeError("No army sets are initialised, call create_army() before")
        else:
            # we cache a copy of the sim as well for convenience
            self.sim_ = simulator.simulate_battle(np.copy(self.M_), self._mapped_ai, **kwargs)
            return self.sim_


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
        if k < 1:
            raise ValueError("'k' must be at least 1")
        if self.M_ is None:
            raise ValueError("Army is not initialised, cannot simulate!")
        else:
            Z = np.zeros((k,2), dtype=np.int64)
            for i in range(k):
                team_counts = simulator.simulate_battle(np.copy(self.M_), self._mapped_ai,
                                                  ret_frames=False, **kwargs)
                Z[i, :] = team_counts
            return pd.DataFrame(Z, columns=self.allegiances_.values)


    """ ------------ CONVENIENCE PLOTTING FUNCTIONS ---------------------- """

    def sim_jupyter(self, func=simplot.quiver_fight, create_html=False, cols={}):
        """
        This convenience method uses any saved 'sim_' object to generate the code
        to output to a Jupyter Notebook. Once must simply then do:

            HTML(battle.sim_jupyter())

        And hey presto, it should all work!

        Parameters
        --------
        func : function
            The plot function to call, by default is bsm.quiver_fight()
        create_html : bool
            Decides whether to return the object directly, or create HTML to then use HTML()
        cols : dict
            colour dictionary to identify each allegiance with.

        Returns
        -------
        s : str/object
            HTML code to feed into HTML(s)
        """
        if self.sim_ is None:
            raise AttributeError("No simulation has occured, no presense of battle.sim_ object.")
        labels = self.allegiances_.to_dict()
        if len(cols) <= 0:
            cols = utils.slice_loop(simplot._loop_colors(), len(self.allegiances_))

        # call plotting function
        Q = func(self.sim_, labels, cols)

        if create_html:
            return Q.to_jshtml()
        else:
            return Q


    def sim_export(self, filename="example_sim.gif",
                   func=simplot.quiver_fight, cols={},
                   writer="pillow"):
        """
        This convenience method uses any saved 'sim_' object to generate the code
        to export into a gif file.

        Parameters
        -------
        filename : str
            The name of the file to output. Must end in .gif
        func : function
            The plot function to call, by default is bsm.quiver_fight()
        cols : dict
            colour dictionary to identify each allegiance with.
        writer : str
            The type of writer to pass to funcanimation.save(). This might
            need to be tweaked on your system.

        Returns
        -------
        None
        """
        # append to end if not present
        if not filename.endswith(".gif"):
            filename.append(".gif")

        if self.sim_ is None:
            raise ValueError("No simulation has occured, no presense of battle.sim_ object.")

        labels = self.allegiances_.to_dict()
        if len(cols) <= 0:
            cols = utils.slice_loop(simplot._loop_colors(), len(self.allegiances_))

        # call function
        Q = func(self.sim_, labels, cols)

        #save
        Q.save(filename,writer=writer)
        return


    """ ---------------------- MISC --------------------------------------- """

    def _get_unit_composition(self):
        if self.M_ is None:
            return None
        d = {}
        d["unit"] = [name for name, _ in self.army_set_]
        d["allegiance"] = [self.db_.loc[u, "Allegiance"] for u,_ in self.army_set_]
        d["n"] = [n for _, n in self.army_set_]
        d["init_ai"] = [a for a in self.init_ai_.values()]
        d["rolling_ai"] = [a for a in self.rolling_ai_.values()]
        return pd.DataFrame(d)


    composition_ = property(_get_unit_composition, doc="The composition of the Battle")