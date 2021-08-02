#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Responsible for creating a Battle object.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Union, Tuple, Callable, List

from . import utils
from . import _target
from .plot import quiver_fight, loop_colors
from . import _ai as AI
from . import _unit_quant as UnitQuant
from .__defaults import default_db
from ._simulator_fast import simulate_battle as sim_battle
from ._distributions import Distribution
from ._terrain import Terrain


class Battle(object):
    """
    This 'Battle' object provides the interface for the user of simulating
    a number of Battles.

    Each simulation follows a:
        Load -> Create -> Simulate -> Draw
    flow.
    """

    """###################### INIT FUNCTION #####################################"""

    def __init__(self,
                 db: Union[str, Dict, pd.DataFrame] = default_db(),
                 bounds: Tuple[float, float, float, float] = (0., 10., 0., 10.)):
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
        # assign with checks
        self.db_ = db
        self._D = []
        self._M = None
        self._sim = None
        self.db_.index = self.db_.index.str.lower()
        # initialise a terrain
        self._T = Terrain(bounds, res=.1, form=None)
        # create internal dictionary called 'army group' or ag
        self._ag = {}

    """####################### HIDDEN FUNCTIONS ##############################"""

    @staticmethod
    def _dataset(n: int):
        return np.zeros(n, dtype=[
            ("team", np.uint8), ("utype", np.uint8), ("pos", np.float32, 2), ("hp", np.float32),
            ("armor", np.float32), ("range", np.float32), ("speed", np.float32), ("acc", np.float32),
            ("dodge", np.float32), ("dmg", np.float32), ("target", np.int32),
            ("group", np.uint8)
        ])

    def _is_instantiated(self):
        if self.M_ is None:
            raise AttributeError("'create_army' has not been called - there are no units.")

    def _is_simulated(self):
        if self.sim_ is None:
            raise AttributeError("No simulation has occurred, no presense of battle.sim_ object.")

    def _determine_target_ai(self, t, pkg=_target):
        self._is_instantiated()
        if isinstance(t, str):
            if t in pkg.get_function_names():
                return [t] * self.n_armies_
            else:
                raise ValueError("value '{}' not found in {}".format(t, pkg.get_function_names()))
        elif isinstance(t, (list, tuple)) and len(t) == self.n_armies_:
            utils.check_in_list(pkg.get_function_names(), t)
            utils.check_list_type(t, str)
            return t
        else:
            raise AttributeError("target ai type must be a type of [str, list, tuple].")

    def _determine_ai_mapping(self, t, pkg=_target):
        mm = pkg.get_map_functions()
        r_armies = range(self.n_armies_)
        army_to_fname = dict(zip(r_armies, t))
        return dict(zip(r_armies, [mm[army_to_fname[f]] for f in army_to_fname]))

    def _plot_simulation(self, func: Callable):
        labels = self.allegiances_.to_dict()
        cols = utils.slice_loop(loop_colors(), len(self.allegiances_))
        # quantify size by value
        qscore = UnitQuant.rank_score(self.db_).reset_index(drop=True).to_dict()

        # call plotting function - with or without terrain
        if self.T_ is not None:
            Q = func(self.sim_, self.T_, labels, cols, qscore)
        else:
            Q = func(self.sim_, None, labels, cols, qscore)
        return Q

    def _initialize_position(self):
        """ Given self.D_, instantiation, we should be able to map positions """
        self._is_instantiated()
        for i, (u, n, (start, end), dist) in enumerate(zip(self._unit_roster, self._unit_n, self._segments, self.D_)):
            self.M_["pos"][start:end] = dist.sample(n)

    def _get_bounds_from_M(self):
        xmin, xmax = self.M_["pos"][:, 0].min(), self.M_["pos"][:, 0].max()
        ymin, ymax = self.M_["pos"][:, 1].min(), self.M_["pos"][:, 1].max()
        return np.floor(xmin), np.ceil(xmax), np.floor(ymin), np.ceil(ymax)

    def _check_bounds_to_M(self, bounds: Tuple[float, float, float, float]):
        xmin, xmax, ymin, ymax = self._get_bounds_from_M()
        if bounds[0] > xmin:
            raise ValueError("xmin bounds value: {} > unit bound {}".format(bounds[0], xmin))
        if bounds[1] < xmax:
            raise ValueError("xmax bounds value: {} < unit bound {}".format(bounds[1], xmax))
        if bounds[2] > ymin:
            raise ValueError("ymin bounds value: {} > unit bound {}".format(bounds[2], ymin))
        if bounds[3] < ymax:
            raise ValueError("ymax bounds value: {} < unit bound {}".format(bounds[3], ymax))

    def _sample_dist(self, D, segments):
        for (u, n), (start, end) in zip(self.army_set_, segments):
            self.D_.append(D)
            self.M_["pos"][start:end] = D.sample(n)

    def _assign_initial_targets(self):

        f_dict = _target.get_global_map_functions()

        for group, (start, end), func, team in zip(range(self.n_armies_), self._segments, self.init_ai_, self._teams):
            mod_func = "global_" + func
            self.M_["target"][start:end] = f_dict[mod_func](
                self.M_["pos"], self.M_["hp"], self.M_["team"], self.M_["group"], group
            )

    """---------------------- PUBLIC ATTRIBUTES AND ATTR METHODS ----------------------------------------"""

    @property
    def composition_(self) -> pd.DataFrame:
        """Determines the composition of the Battle."""
        self._is_instantiated()
        d = {"unit": [name for name, _ in self.army_set_],
             "allegiance": [self.db_.loc[u, "Allegiance"] for u, _ in self.army_set_],
             "n": [n for _, n in self.army_set_],
             "position": ["{} ({:0.1f}, {:0.1f})".format(dist.dist_, dist.mean_[0], dist.mean_[1]) for dist in self.D_],
             "init_ai": self.init_ai_,
             "rolling_ai": self.rolling_ai_,
             "decision_ai": self.decision_ai_}
        return pd.DataFrame(d)

    @property
    def n_allegiance_(self):
        """Determines the number of teams present in the fight."""
        self._is_instantiated()
        d = {"allegiance": [self.db_.loc[u, "Allegiance"] for u, _ in self.army_set_],
             "n": [n for _, n in self.army_set_]}
        return pd.DataFrame(d).groupby("allegiance")["n"].sum()

    @property
    def bounds_(self) -> Tuple[float, float, float, float]:
        """Determine the bounds of the fight using the Terrain."""
        return self.T_.bounds_

    @bounds_.setter
    def bounds_(self, b: Tuple[float, float, float, float]):
        if self.M_ is None:
            warnings.warn("bounds {} set before units are initialised - some may be out-of-bounds".format(b),
                          UserWarning)
            self.T_.bounds_ = b
        else:
            self._check_bounds_to_M(b)
            self.T_.bounds_ = b

    @property
    def M_(self) -> np.ndarray:
        """The raw data set underlying."""
        return self._M

    @property
    def sim_(self):
        """The simulation object."""
        return self._sim

    @property
    def db_(self) -> pd.DataFrame:
        """The datafiles storing information on each unit."""
        return self._db

    @db_.setter
    def db_(self, db_n):
        if isinstance(db_n, str):
            self._db = utils.import_and_check_unit_file(db_n)
        elif isinstance(db_n, dict):
            self._db = pd.DataFrame(db_n)
            utils.check_unit_file(self._db)
            utils.preprocess_unit_file(self._db)
        elif isinstance(db_n, pd.DataFrame):
            self._db = db_n.copy()
            utils.check_unit_file(self._db)
            utils.preprocess_unit_file(self._db)
        else:
            raise ValueError("'db' must be of type [str, dict, pd.DataFrame], not {}".format(type(db_n)))

    @property
    def T_(self) -> Terrain:
        """Attribute to the Terrain object."""
        return self._T

    @property
    def army_set_(self):
        """A list of unit rosters and counts."""
        self._is_instantiated()
        return tuple(zip(self._unit_roster, self._unit_n))

    @property
    def n_armies_(self) -> int:
        """The number of army types."""
        self._is_instantiated()
        return len(self._unit_roster)

    @property
    def N_(self) -> int:
        """The total number of units."""
        self._is_instantiated()
        return sum(self._unit_n)

    @property
    def D_(self):
        """A set of Distribution objects."""
        return self._D

    @D_.setter
    def D_(self, d):
        """ where d is a list/tuple of distribution, str, or dict """
        self._is_instantiated()
        self._D = []
        if isinstance(d, (str, Distribution)):
            self._D = [Distribution(d) for _ in range(self.n_armies_)]
        elif isinstance(d, dict):
            self._D = [Distribution(**d) for _ in range(self.n_armies_)]
        elif isinstance(d, (list, tuple)):
            for dist in d:
                if isinstance(dist, (str, Distribution)):
                    self._D.append(Distribution(dist))
                elif isinstance(dist, dict):
                    self._D.append(Distribution(**dist))
        else:
            raise TypeError("'d' must be of type [list, tuple, str, Distribution, dict]")

    @property
    def allegiances_(self):
        """The list of allegiances."""
        return self.db_[["Allegiance", "allegiance_int"]].set_index("allegiance_int").drop_duplicates().squeeze()

    @property
    def init_ai_(self):
        """The initialization AI chosen for selecting targets."""
        return self._init_ai

    @init_ai_.setter
    def init_ai_(self, ia):
        self._init_ai = self._determine_target_ai(ia)

    @property
    def rolling_ai_(self):
        """The AI chosen whenever an enemy unit dies."""
        return self._rolling_ai

    @rolling_ai_.setter
    def rolling_ai_(self, ra):
        self._rolling_ai = self._determine_target_ai(ra)

    @property
    def decision_ai_(self):
        """The AI chosen to govern the actions of each army set."""
        return self._decision_ai

    @decision_ai_.setter
    def decision_ai_(self, da):
        self._decision_ai = self._determine_target_ai(da, pkg=AI)

    ################### HIDDEN ATTRIBUTE #################################

    @property
    def _segments(self) -> Tuple[str, int]:
        return utils.get_segments(self.army_set_)

    @property
    def _teams(self) -> np.ndarray:
        self._is_instantiated()
        return np.asarray([self.db_.loc[u, "allegiance_int"] for u in self._unit_roster])

    """--------------------------------- PUBLIC FUNCTIONS ------------------------------------------------"""

    def create_army(self, army_set: Tuple[str, int]):
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
        # assign
        utils.is_twotuple(army_set, str, (int, np.int, np.int64))
        # assign unit roster, n for roster
        self._unit_roster = [u.lower() for u, _ in army_set]
        self._unit_n = [n for _, n in army_set]

        self._M = Battle._dataset(sum(self._unit_n))
        # check that groups exist in army_set
        utils.check_groups_in_db(self.army_set_, self.db_)

        # capture distributions
        self._D = [Distribution("normal", loc=i * 2., scale=1.) for i in range(self.n_armies_)]

        # set initial values.
        for i, (u, n, (start, end)) in enumerate(zip(self._unit_roster, self._unit_n, self._segments)):
            self.M_["utype"][start:end] = np.argwhere(self.db_.index == u).flatten()[0]
            self.M_["team"][start:end] = self.db_.loc[u, "allegiance_int"]
            self.M_["group"][start:end] = i
            self.M_["hp"][start:end] = self.db_.loc[u, "HP"]
            self.M_["armor"][start:end] = self.db_.loc[u, "Armor"]
            self.M_["range"][start:end] = self.db_.loc[u, "Range"]
            self.M_["speed"][start:end] = self.db_.loc[u, "Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u, "Miss"] / 100.
            self.M_["acc"][start:end] = self.db_.loc[u, "Accuracy"] / 100.
            self.M_["dmg"][start:end] = self.db_.loc[u, "Damage"]

        self._initialize_position()

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
        # assign and check
        self.D_ = distributions
        # assign positions
        self._initialize_position()
        # modify bounds
        self.bounds_ = self._get_bounds_from_M()
        return self

    def apply_terrain(self, t=None, res: float = .1):
        """
        Applies a Z-plane to the map that the Battle is occuring on by creating
        a bsm.Terrain object.

        Parameters
        -------
        t : str or bsm.Terrain
            Choose from [None, 'grid', 'contour']. Default is None. Contour looks
            the best. Decides how big/resolution to make the terrain based on the
            initialized positions of units.
        res : float
            The resolution to use for the map

        Returns
        -------
        self
        """
        self._is_instantiated()

        if t in [None, "grid", "contour"]:
            # add function to t
            self.T_.res_ = res
            self.T_.form_ = t
            return self
        elif isinstance(t, Terrain):
            self._T = t
            return self
        else:
            raise ValueError("'t' must be [grid, contour, None]")

    def set_initial_ai(self, func_names: Union[str, List[str]]):
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
        self.init_ai_ = func_names
        return self

    def set_rolling_ai(self, func_names: Union[str, List[str]]):
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
        self.rolling_ai_ = func_names
        # map these strings to actual functions, ready for simulate.
        self._rolling_map = self._determine_ai_mapping(self.rolling_ai_)
        return self

    def set_decision_ai(self, decision: Union[str, List[str]]):
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
        self.decision_ai_ = decision
        # map these strings to actual functions, ready for simulate.
        self._decision_map = self._determine_ai_mapping(self.decision_ai_, pkg=AI)
        return self

    def set_bounds(self, bounds: Tuple[float, float, float, float]):
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
        if np.unique(self._teams).shape[0] <= 1:
            warnings.warn("Simulation halted - There is only one team present.", UserWarning)
            return self.sim_
        # set the flat terrain if it doesn't exist
        self.T_.generate()
        # firstly assign initial AI targets.
        self._assign_initial_targets()
        # we cache a copy of the sim as well for convenience
        self._sim = sim_battle(np.copy(self.M_),
                               self.T_,
                               self._rolling_map,
                               self._decision_map,
                               ret_frames=True,
                               **kwargs)
        return self.sim_

    def simulate_k(self, k: int = 10, **kwargs):
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
        runs : pd.DataFrame
            The iteration (k), with the team victorious, and number of units remaining

        Returns the victory for each k iteration, for each team.
        """
        self._is_instantiated()

        if k < 1:
            raise ValueError("'k' must be at least 1")
        else:
            # check for multiple teams
            if np.unique(self._teams).shape[0] <= 1:
                warnings.warn("Simulation halted - There is only one team present.", UserWarning)
                return self.sim_

            runs = np.zeros((k, 2), dtype=np.int64)
            # generate new terrain
            self.T_.generate()
            for i in range(k):
                # firstly assign initial AI targets.
                self._assign_initial_targets()
                # run simulation
                team_counts = sim_battle(np.copy(self.M_),
                                         self.T_,
                                         self._rolling_map,
                                         self._decision_map,
                                         ret_frames=False,
                                         **kwargs)
                runs[i, :] = team_counts
            return pd.DataFrame(runs, columns=self.allegiances_.values)

    """ ------------ CONVENIENCE PLOTTING FUNCTIONS ---------------------- """

    def sim_jupyter(self,
                    func: Callable = quiver_fight,
                    create_html: bool = False):
        """
        This convenience method uses any saved 'sim_' object to generate the code
        to output to a Jupyter Notebook. Once must simply then do:

            HTML(battle.sim_jupyter())

        And hey presto, it should all work!

        Parameters
        --------
        func : function, optional
            The plot function to call, by default is bsm.quiver_fight()
        create_html : bool, optional
            Decides whether to return the object directly, or create HTML to then use HTML()

        Returns
        -------
        s : str/object
            HTML code to feed into HTML(s)
        """
        self._is_simulated()
        # call plotting function - with
        Q = self._plot_simulation(func)

        if create_html:
            return Q.to_jshtml()
        else:
            return Q

    def sim_export(self,
                   filename: str = "example_sim.gif",
                   func: Callable = quiver_fight,
                   writer: str = "pillow"):
        """
        This convenience method uses any saved 'sim_' object to generate the code
        to export into a gif file.

        Parameters
        -------
        filename : str, optional
            The name of the file to output. Must end in .gif
        func : function, optional
            The plot function to call, by default is bsm.quiver_fight()
        writer : str, optional
            The type of writer to pass to funcanimation.save(). This might
            need to be tweaked on your system.
            Accepts ['imagemagick', 'ffmpeg', 'pillow']

        Returns
        -------
        None
        """
        self._is_simulated()
        # append to end if not present
        if not filename.endswith(".gif"):
            filename.append(".gif")

        # call simulation
        Q = self._plot_simulation(func)

        # save
        Q.save(filename, writer=writer)
        return

    """ ---------------------- MISC --------------------------------------- """

    def __repr__(self) -> str:
        if self.M_ is None:
            return "bsm.Battle(init=False)"
        elif self.sim_ is None:
            return "bsm.Battle(init=True, n_armies={}, simulated=False)".format(self.n_armies_)
        else:
            return "bsm.Battle(init=True, n_armies={}, simulated=True)".format(self.n_armies_)
