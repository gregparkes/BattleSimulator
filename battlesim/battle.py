#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:39:42 2019

@author: gparkes
"""

import numpy as np
import pandas as pd
import warnings

from . import utils
from . import target
from . import simplot
from . import ai
from . import unit_quant
from . import defaults
from .simulator_fast import simulate_battle as sim_battle
from .distributions import Distribution
from .terrain import Terrain


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
            raise AttributeError("'create_army' has not been called - there are no units.")


    def _is_simulated(self):
        if self.sim_ is None:
            raise AttributeError("No simulation has occured, no presense of battle.sim_ object.")


    def _plot_simulation(self, func, cols):
        labels = self.allegiances_.to_dict()
        if len(cols) <= 0:
            cols = utils.slice_loop(simplot._loop_colors(), len(self.allegiances_))
        # quantify size by value
        qscore = unit_quant.rank_score(self.db_).reset_index(drop=True).to_dict()

        # call plotting function - with or without terrain
        if self.T_ is not None:
            Q = func(self.sim_, self.T_, labels, cols, qscore)
        else:
            Q = func(self.sim_, None, labels, cols, qscore)
        return Q


    def _get_bounds_from_M(self):
        xmin, xmax = self.M_["pos"][:, 0].min(), self.M_["pos"][:, 0].max()
        ymin, ymax = self.M_["pos"][:, 1].min(), self.M_["pos"][:, 1].max()
        return np.floor(xmin), np.ceil(xmax), np.floor(ymin), np.ceil(ymax)


    def _check_bounds_to_M(self, bounds):
        xmin, xmax, ymin, ymax = self._get_bounds_from_M()
        if bounds[0] > xmin:
            raise ValueError("xmin bounds value: {} > unit bound {}".format(bounds[0],xmin))
        if bounds[1] < xmax:
            raise ValueError("xmax bounds value: {} < unit bound {}".format(bounds[1],xmax))
        if bounds[2] > ymin:
            raise ValueError("ymin bounds value: {} > unit bound {}".format(bounds[2],ymin))
        if bounds[3] < ymax:
            raise ValueError("ymax bounds value: {} < unit bound {}".format(bounds[3],ymax))


    def _sample_dist(self, D, segments):
        for (u, n), (start, end) in zip(self.army_set_, segments):
            self.D_.append(D)
            self.M_["pos"][start:end] = D.sample(n)


    def _assign_initial_targets(self, init_ai):

        self._is_instantiated()

        utils.check_in_list(target.get_init_function_names(), init_ai)
        utils.check_list_type(init_ai, str)

        f_dict = target.get_global_map_functions()
        segments = utils.get_segments(self.army_set_)

        for group, (u, n), (start, end), func, team in zip(range(len(init_ai)), self.army_set_, segments, init_ai, self.teams_):
            mod_func = "global_" + func
            self.M_["target"][start:end] = f_dict[mod_func](
                self.M_["pos"], self.M_["hp"], self.M_["team"], self.M_["group"], group
            )



    ###################### INIT FUNCTION #####################################

    def __init__(self, db=defaults.default_db(), bounds=(0, 10, 0, 10)):
        """
        Instantiate this object with a filepath leading to

        Parameters
        -------
        db : str, dict or pandas.DataFrame
            If str: Is filepath to the database object
            If dict or pandas.dataFrame: represents actual data.
            Must contain ["Name", "Allegiance", "HP", "Damage", "Accuracy", "Miss", "Movement Speed", "Range"] columns.
            See bsm.defaults.default_db() for example.
        bounds : tuple (4,)
            The left, right, top and bottom bounds of the battle. Units cannot
            leave these bounds.
        """
        assert isinstance(bounds, tuple), "bounds must be a tuple"
        assert len(bounds) == 4, "bounds must be of length 4"

        if isinstance(db, str):
            self.db_ = utils.import_and_check_unit_file(db)
        elif isinstance(db, dict):
            self.db_ = pd.DataFrame(db)
            utils.check_unit_file(self.db_)
            utils.preprocess_unit_file(self.db_)
        elif isinstance(db, pd.DataFrame):
            self.db_ = db.copy()
            utils.check_unit_file(self.db_)
            utils.preprocess_unit_file(self.db_)
        else:
            raise ValueError("'db' must be of type [str, dict, pd.DataFrame], not {}".format(type(db)))

        self.M_ = None
        self.sim_ = None
        # convert db_ index to lower case.
        self.db_names_ = self.db_.index.tolist()
        self.db_.index = self.db_.index.str.lower()
        # initialise a terrain
        self.T_ = Terrain(bounds, res=.1, form=None)


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

        # capture distributions
        self.D_ = []

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
            self.D_.append(D)
            self.M_["pos"][start:end] = D.sample(n)

        # modify bounds
        self.bounds_ = self._get_bounds_from_M()

        # initialise AIs as nearest.
        self.set_initial_ai("nearest")
        self.set_rolling_ai("nearest")
        # main AI options
        self.set_decision_ai("aggressive")

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

        self.D_ = []

        segments = utils.get_segments(self.army_set_)
        if isinstance(distributions, str):
            self._sample_dist(Distribution(distributions), segments)
        elif isinstance(distributions, Distribution):
            self._sample_dist(distributions, segments)
        elif isinstance(distributions, dict):
            self._sample_dist(Distribution(**distributions), segments)
        elif isinstance(distributions, (list, tuple)):
            for (u, n), (start, end), d in zip(self.army_set_, segments, distributions):
                if isinstance(d, Distribution):
                    self.D_.append(d)
                    self.M_["pos"][start:end] = d.sample(n)
                elif isinstance(d, dict):
                    # unpack keywords into the 'dist' function of Distribution
                    D = Distribution(**d)
                    self.D_.append(D)
                    self.M_["pos"][start:end] = D.sample(n)
                elif isinstance(d, str):
                    # each is a string distribution
                    D = Distribution(d)
                    self.D_.append(D)
                    self.M_["pos"][start:end] = D.sample(n)
                else:
                    raise TypeError("Each element of 'distributions' must be a bsm.Distribution or dict")
        else:
            raise TypeError("distributions must be of type [str, Distribution, list, tuple]")

        # modify bounds
        self.bounds_ = self._get_bounds_from_M()
        return self


    def apply_terrain(self, t=None, res=.1, f=None):
        """
        Applies a Z-plane to the map that the Battle is occuring on by creating
        a bsm.Terrain object.

        Parameters
        -------
        t : str
            Choose from [None, 'grid', 'contour']. Default is None. Contour looks
            the best. Decides how big/resolution to make the terrain based on the
            initialized positions of units.
        f : function
            A function z = f(x, y) that performs a mathematical transformation, or None
            for random hills.

        Returns
        -------
        self
        """
        self._is_instantiated()

        if t in [None, "grid", "contour"]:
            # add function to t
            self.T_.res_ = res
            self.T_.form_ = t
            # apply
            self.T_.generate(f=f)
            return self
        else:
            raise ValueError("'t' must be [grid, contour, None]")


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
            if func_names not in target.get_init_function_names():
                raise ValueError("init_ai '{}' not found in {}".format(func_names, target.get_init_function_names()))
            self.init_ai_ = func_names = [func_names] * self.n_armies_
        elif isinstance(func_names, (list, tuple)) and (len(func_names) == self.n_armies_):
            utils.check_in_list(target.get_init_function_names(), func_names)
            self.init_ai_ = func_names
        else:
            raise AttributeError("ai_funcs is wrong type or length.")
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
            if func_names not in target.get_init_function_names():
                raise ValueError("init_ai '{}' not found in {}".format(func_names, target.get_init_function_names()))
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


    def set_decision_ai(self, decision):
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
        return self


    def set_bounds(self, bounds):
        """
        Sets the boundaries of the Battle. If not initialised, this is OK but may
        produce errors down-the-line.

        Parameters
        -------
        bounds : tuple, list (4,)
            the (left, right, top, bottom) dimensions of the simulation.

        Returns
        -------
        self
        """
        self.bounds_ = bounds
        return self

    """ ----------------------------- SIMULATION ----------------------------- """

    def simulate(self, **kwargs):
        """
        Runs the 'simulate_battle' algorithm. Creates and passes a copy to simulate..

        Returns pd.DataFrame of frames.
        """
        self._is_instantiated()
        # check for multiple teams
        if np.unique(self.teams_).shape[0] <= 1:
            warnings.warn("Simulation halted - There is only one team present.", UserWarning)
            return self.sim_
        # set the flat terrain if it doesn't exist
        self.T_.generate()
        # firstly assign initial AI targets.
        self._assign_initial_targets(self.init_ai_)
        # we cache a copy of the sim as well for convenience
        self.sim_ = sim_battle(np.copy(self.M_),
                             self.T_,
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
            # check for multiple teams
            if np.unique(self.teams_).shape[0] <= 1:
                warnings.warn("Simulation halted - There is only one team present.", UserWarning)
                return self.sim_

            Z = np.zeros((k,2), dtype=np.int64)
            # generate new terrain
            self.T_.generate()
            for i in range(k):
                # firstly assign initial AI targets.
                self._assign_initial_targets(self.init_ai_)
                # run simulation
                team_counts = sim_battle(np.copy(self.M_),
                                       self.T_,
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
        d["position"] = ["{} ({:0.1f}, {:0.1f})".format(dist.dist_, dist.mean_[0], dist.mean_[1]) for dist in self.D_]
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

    def _get_bounds(self):
        return self.T_.bounds_

    def _set_bounds(self, b):
        utils.is_ntuple(b, (float, int), (float, int), (float, int), (float, int))
        if self.M_ is None:
            warnings.warn("bounds {} set before units are initialised - some may be out-of-bounds".format(b), UserWarning)
            self.T_.bounds_ = b
        else:
            self._check_bounds_to_M(b)
            self.T_.bounds_ = b

    bounds_ = property(_get_bounds, _set_bounds, doc="bounds of the battle")
    composition_ = property(_get_unit_composition, doc="The composition of the Battle")
    n_allegiance_ = property(_get_n_allegiance, doc="get the number of units for each side")

    def __repr__(self):
        if self.M_ is None:
            return "bsm.Battle(init=False)"
        elif self.sim_ is None:
            return "bsm.Battle(init=True, n_armies={}, simulated=False)".format(self.n_armies_)
        else:
            return "bsm.Battle(init=True, n_armies={}, simulated=True)".format(self.n_armies_)