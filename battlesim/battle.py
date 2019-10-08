#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:39:42 2019

@author: gparkes
"""

import numpy as np
import pandas as pd

from . import utils
from .simulator_fast import simulate_battle as sim_battle
from . import target
from . import simplot
from . import ai
from .distributions import dist, Distribution, get_options
from . import unit_quant


class Battle(object):
    """
    This 'Battle' object provides the interface for the user of simulating
    a number of Battles.

    Each simulation follows a:
        Load -> Create -> Simulate -> Draw
    flow.
    """

    ####################### HIDDEN FUNCTIONS ##############################


    def _dataset(self, n):
        return np.zeros((n), dtype=[
            ("team",np.uint8,1),("utype",np.uint8,1),("pos",np.float64,2),("hp",np.float64,1),
            ("range",np.float64,1),("speed",np.float64,1),("acc",np.float64,1),
            ("dodge",np.float64,1),("dmg",np.float64,1),("target",np.int64,1),
            ("group",np.uint8,1)
        ])


    def _is_instantiated(self):
        if self.M_ is None:
            raise TypeError("'create_army' has not been called - there are no units.")


    def _is_simulated(self):
        if self.sim_ is None:
            raise AttributeError("No simulation has occured, no presense of battle.sim_ object.")


    def _plot_simulation(self, func, cols):
        labels = self.allegiances_.to_dict()
        if len(cols) <= 0:
            cols = utils.slice_loop(simplot._loop_colors(), len(self.allegiances_))

        # quantify size by value
        qscore = unit_quant.rank_score(self.db_).reset_index(drop=True).to_dict()

        # call plotting function - with
        Q = func(self.sim_, labels, cols, qscore)
        return Q


    def _assign_initial_targets(self, init_ai):

        self._is_instantiated()

        utils.check_in_list(target.get_init_function_names(), init_ai)
        utils.check_list_type(init_ai, str)

        f_dict = target.get_map_functions()
        segments = utils.get_segments(self.army_set_)

        # create valid_targets set
        valid_targets = [np.argwhere((self.M_["team"]!=T)).flatten() for T in np.unique(self.M_["team"])]
        valid_allies = [np.argwhere((self.M_["team"]==T)).flatten() for T in np.unique(self.M_["team"])]

        for (u, n), (start, end), func, team in zip(self.army_set_, segments, init_ai, self.teams_):
            for i in range(start, end):
                # AI template <pos>,<target>,<hp>,<enemies>,<allies>,<index>
                self.M_["target"][i] = f_dict[func](self.M_["pos"],
                       self.M_["target"],
                       self.M_["hp"],
                       valid_targets[team],
                       valid_allies[team],
                       i)


    ###################### INIT FUNCTION #####################################

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

        self.M_ = self._dataset(self.N_)

        segments = utils.get_segments(army_set)
        self.teams_ = np.asarray([self.db_.loc[U[0],"allegiance_int"] for U in self.army_set_])
        self.allegiances_ = (self.db_[["Allegiance","allegiance_int"]]
                                .set_index("allegiance_int")
                                .drop_duplicates().squeeze())

        # assign a 'distribution' name
        self.dname_ = []

        # set initial values.
        for i, ((u, n), (start, end), team) in enumerate(zip(self.army_set_, segments, self.teams_)):
            self.M_["team"][start:end] = team
            self.M_["utype"][start:end] = np.argwhere(self.db_.index == u).flatten()[0]
            self.M_["group"][start:end] = i
            self.M_["hp"][start:end] = self.db_.loc[u,"HP"]
            self.M_["range"][start:end] = self.db_.loc[u,"Range"]
            self.M_["speed"][start:end] = self.db_.loc[u,"Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u,"Miss"]/100.
            self.M_["acc"][start:end] = self.db_.loc[u,"Accuracy"]/100.
            self.M_["dmg"][start:end] = self.db_.loc[u,"Damage"]
            # default position: uniform
            D = Distribution('normal', loc=i*2., scale=1.)
            self.dname_.append('normal')
            self.M_["pos"][start:end] = D.sample(n)

        # initialise AIs as nearest.
        self.set_initial_ai("nearest")
        self.set_rolling_ai("nearest")
        # main AI options
        self.set_ai_decision("aggressive")

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
        self._is_instantiated()

        self.dname_ = []

        segments = utils.get_segments(self.army_set_)
        if isinstance(distributions, str):
            if distributions in get_options()["names"]:
                # create a distribution object.
                D = Distribution(distributions)
                for (u, n), (start, end) in zip(self.army_set_, segments):
                    self.dname_.append(D.dist_)
                    self.M_["pos"][start:end] = D.sample(n)
            else:
                raise ValueError("distribution '{}' not found in bsm.Distribution.".format(distributions))
        elif isinstance(distributions, Distribution):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                self.dname_.append(distributions.dist_)
                self.M_["pos"][start:end] = distributions.sample(n)
        elif isinstance(distributions, dict):
            for (u, n), (start, end) in zip(self.army_set_, segments):
                D = dist(**distributions)
                self.dname_.append(D.dist_)
                self.M_["pos"][start:end] = D.sample(n)
        elif isinstance(distributions, (list, tuple)):
            for (u, n), (start, end), d in zip(self.army_set_, segments, distributions):
                if isinstance(d, Distribution):
                    self.dname_.append(d.dist_)
                    self.M_["pos"][start:end] = d.sample(n)
                elif isinstance(d, dict):
                    # unpack keywords into the 'dist' function of Distribution
                    D = dist(**d)
                    self.dname_.append(D.dist_)
                    self.M_["pos"][start:end] = D.sample(n)
                elif isinstance(d, str):
                    # each is a string distribution
                    if d in get_options()["names"]:
                        # create a distribution object.
                        D = Distribution(d)
                        self.dname_.append(D.dist_)
                        self.M_["pos"][start:end] = D.sample(n)
                    else:
                        raise ValueError("distribution '{}' not found in bsm.Distribution.".format(d))
                else:
                    raise TypeError("Each element of 'distributions' must be a bsm.Distribution or dict")
        else:
            raise TypeError("distributions must be of type [str, Distribution, list, tuple]")
        return self


    @utils.to_remove
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
        self._is_instantiated()

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
        func_names : str or list of str
            The AI function to choose for initial targets for each army set. Can choose from:
                ['random', 'pack', 'nearest']

        Returns
        -------
        self
        """
        self._is_instantiated()

        if isinstance(func_names, str):
            # set the string for all.
            self.init_ai_ = func_names = [func_names] * self.n_armies_
        elif isinstance(func_names, (list, tuple)) and (len(func_names) == self.n_armies_):
            self.init_ai_ = func_names
        else:
            raise AttributeError("ai_funcs is wrong type or length.")

        self._assign_initial_targets(func_names)

        return self


    def set_rolling_ai(self, func_names):
        """
        Set the AI decisions for choosing a target rolling through the simulation.
        Pass a function name from bsm.ai for each 'army set'.

        Parameters
        --------
        func_names : str or list of str
            The AI function to choose for targets when their target dies
            for each army set. Can choose from: ['random', 'pack', 'nearest']

        Returns
        -------
        self
        """
        self._is_instantiated()

        if isinstance(func_names, str):
            self.rolling_ai_ = func_names = [func_names]*self.n_armies_
        elif isinstance(func_names, (list, tuple)):
            self.rolling_ai_ = func_names

        utils.check_in_list(target.get_init_function_names(), func_names)
        utils.check_list_type(func_names, str)

        # map these strings to actual functions, ready for simulate.
        mappp = target.get_map_functions()
        army_to_fname = dict(zip(range(self.n_armies_), func_names))

        self._rolling_map = dict(zip(range(self.n_armies_), [mappp[army_to_fname[f]] for f in army_to_fname]))
        return self


    def set_ai_decision(self, decision):
        """
        Sets the over-arching AI choices for each 'army group'. By default they
        choose an 'aggressive' stance.

        WARNING: This function CAN override the choices made for the 'initial'
        and 'rolling' AI decisions, based on it's functionality. This function
        is therefore superior to those options.

        Parameters
        --------
        decision : str or list of str
            If str, applies that option to all army groups. Choose from
            [aggressive, hit_and_run]

        Returns
        -------
        self
        """
        if isinstance(decision, str):
            self.decision_ai_ = decision = [decision] * self.n_armies_
        if isinstance(decision, (list, tuple)):
            utils.check_in_list(ai.get_function_names(), decision)
            utils.check_list_type(decision, str)
            self._is_instantiated()

            self.decision_ai_ = decision
            mappp = ai.get_function_map()
            army_to_dname = dict(zip(range(self.n_armies_), decision))
            # map id to function
            self._decision_map = dict(zip(range(self.n_armies_),
                                          [mappp[army_to_dname[f]] for f in army_to_dname]))

        else:
            raise TypeError("'ai' must be [str, list, tuple]")


    @utils.to_remove
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


    @utils.to_remove
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
        self._is_instantiated()

        # firstly assign initial AI targets.
        self._assign_initial_targets(self.init_ai_)
        # we cache a copy of the sim as well for convenience
        self.sim_ = sim_battle(np.copy(self.M_),
                             self._rolling_map,
                             self._decision_map,
                             ret_frames=True,
                             **kwargs)
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
        self._is_instantiated()

        if k < 1:
            raise ValueError("'k' must be at least 1")
        else:
            Z = np.zeros((k,2), dtype=np.int64)
            for i in range(k):
                # firstly assign initial AI targets.
                self._assign_initial_targets(self.init_ai_)
                # run simulation
                team_counts = sim_battle(np.copy(self.M_),
                                       self._rolling_map,
                                       self._decision_map,
                                       ret_frames=False,
                                       **kwargs)
                Z[i, :] = team_counts
            return pd.DataFrame(Z, columns=self.allegiances_.values)


    """ ------------ CONVENIENCE PLOTTING FUNCTIONS ---------------------- """


    def sim_jupyter(self,
                    func=simplot.quiver_fight,
                    create_html=False,
                    cols={}):
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
        self._is_simulated()
        # call plotting function - with
        Q = self._plot_simulation(func, cols)

        if create_html:
            return Q.to_jshtml()
        else:
            return Q


    def sim_export(self,
                   filename="example_sim.gif",
                   func=simplot.quiver_fight,
                   cols={},
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
        self._is_simulated()
        # append to end if not present
        if not filename.endswith(".gif"):
            filename.append(".gif")

        # call simulation
        Q = self._plot_simulation(func, cols)

        #save
        Q.save(filename,writer=writer)
        return


    """ ---------------------- MISC --------------------------------------- """

    def _get_unit_composition(self):
        self._is_instantiated()
        d = {}
        d["unit"] = [name for name, _ in self.army_set_]
        d["allegiance"] = [self.db_.loc[u, "Allegiance"] for u,_ in self.army_set_]
        d["n"] = [n for _, n in self.army_set_]
        d["position"] = self.dname_
        d["init_ai"] = self.init_ai_
        d["rolling_ai"] = self.rolling_ai_
        d["decision_ai"] = self.decision_ai_
        return pd.DataFrame(d)


    def _get_n_allegiance(self):
        self._is_instantiated()
        d = {}
        d["allegiance"] = [self.db_.loc[u, "Allegiance"] for u,_ in self.army_set_]
        d["n"] = [n for _, n in self.army_set_]
        return pd.DataFrame(d).groupby("allegiance")["n"].sum()


    composition_ = property(_get_unit_composition, doc="The composition of the Battle")
    n_allegiance_ = property(_get_n_allegiance, doc="get the number of units for each side")

    def __repr__(self):
        if self.M_ is None:
            return "bsm.Battle(init=False)"
        elif self.sim_ is None:
            return "bsm.Battle(init=True, n_armies={}, simulated=False)".format(self.n_armies_)
        else:
            return "bsm.Battle(init=True, n_armies={}, simulated=True)".format(self.n_armies_)