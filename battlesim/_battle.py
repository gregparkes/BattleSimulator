#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Responsible for creating a Battle object.
"""

import itertools as it
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import colors

from battlesim.terra import Terrain
from . import _utils
from .__defaults import default_db
from .distrib import Composite
from .plot import quiver_fight
from .simulation import _ai as AI, _target, simulate_battle as sim_battle


class Battle:
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
                 bounds: Tuple[float, float, float, float] = (0., 10., 0., 10.),
                 use_tqdm: bool = True):
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
        use_tqdm : bool, default=True
            Draws a progressbar with `simulate_k` if tqdm is installed
        """
        self.use_tqdm = use_tqdm
        # assign with checks
        self.db_ = db
        self._M = None
        self._S = None
        self._sim = None
        self.db_.index = self.db_.index.str.lower()
        # initialise a terra
        self._T = Terrain(bounds, res=.1, form=None)
        # design a list of composites
        self._comps = []
        self._decision_map = {"aggressive": 0, "hit_and_run": 1}


    """####################### HIDDEN FUNCTIONS ##############################"""

    @staticmethod
    def _generate_M(n: int):
        return np.zeros(n, dtype=np.dtype([
            ("id", "u4"), ("target", "u4"), ("x", "f4"), ("y", "f4"),
            ("hp", "f4"), ("armor", "f4"), ("dmg", "f4"), ("range", "f4"), ("speed", "f4"),
            ("acc", "f4"), ("dodge", "f4"),
            ("utype", "u1"), ("team", "u1"), ("ai_func_index", "u1")
        ], align=True))

    def _loading_bar(self, k: int):
        if self.use_tqdm and _utils.is_tqdm_installed(False):
            from tqdm import tqdm
            return tqdm(range(k))
        else:
            return range(k)

    def _is_instantiated(self):
        if self._comps is None:
            raise AttributeError("'create_army' has not been called - there are no units.")

    def _is_simulated(self):
        if self.sim_ is None:
            raise AttributeError("No simulation has occurred, no presense of battle.sim_ object.")

    def _plot_simulation(self, func: Callable):
        labels = self.allegiances_.to_dict()
        cols = _utils.slice_loop(colors.BASE_COLORS.keys(), len(self.allegiances_))
        # call plotting function - with or without terra
        if self.T_ is not None:
            Q = func(self.sim_, self.T_, labels, cols)
        else:
            Q = func(self.sim_, None, labels, cols)
        return Q

    def _get_bounds_from_M(self):
        xmin, xmax = self.M_["x"].min(), self.M_["x"].max()
        ymin, ymax = self.M_["y"].min(), self.M_["y"].max()
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

    def _presim(self):
        self._M = Battle._generate_M(sum(self._unit_n))
        # check that groups exist in army_set
        _seg_start, _seg_end = self._segments
        decision_ai_map = dict(zip(AI.get_function_names(), it.count()))

        # set initial values.
        for group, (u, n, start, end, comp) in enumerate(zip(self._unit_roster, self._unit_n, _seg_start, _seg_end, self._comps)):
            # set mutable M values in larger matrix.
            self.M_['hp'][start:end] = self.db_.loc[u, "HP"]
            self.M_["armor"][start:end] = self.db_.loc[u, "Armor"]
            self.M_['team'][start:end] = self.db_.loc[u, "allegiance_int"]
            self.M_["id"][start:end] = group
            self.M_['utype'][start:end] = np.argwhere(self.db_.index == u).flatten()[0]
            self.M_["range"][start:end] = self.db_.loc[u, "Range"]
            self.M_["speed"][start:end] = self.db_.loc[u, "Movement Speed"]
            self.M_["dodge"][start:end] = self.db_.loc[u, "Miss"] / 100.
            self.M_["acc"][start:end] = self.db_.loc[u, "Accuracy"] / 100.
            self.M_["dmg"][start:end] = self.db_.loc[u, "Damage"]
            # ai func index (0 = aggressive, 1 = hit_and_run)
            self.M_['ai_func_index'][start:end] = decision_ai_map[comp.decision_ai]
            # initialise position
            self.M_["x"][start:end] = comp.pos.sample(n)
            self.M_["y"][start:end] = comp.pos.sample(n)

        # modify bounds to reflect new positions.
        self.bounds_ = self._get_bounds_from_M()
        # assign initial AI targets.
        for group, (start, end) in enumerate(zip(_seg_start, _seg_end)):
            # assign targets
            self.M_["target"][start:end] = _target.global_nearest(self.M_, group)

    """---------------------- PUBLIC ATTRIBUTES AND ATTR METHODS ----------------------------------------"""

    @property
    def composition_(self):
        """Determines the composition of the Battle."""
        self._is_instantiated()
        return self._comps

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
        self._is_instantiated()
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
        """The mutable (updatable) matrix information."""
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
            self._db = _utils.import_and_check_unit_file(db_n)
        elif isinstance(db_n, dict):
            self._db = pd.DataFrame(db_n)
            _utils.check_unit_file(self._db)
            _utils.preprocess_unit_file(self._db)
        elif isinstance(db_n, pd.DataFrame):
            self._db = db_n.copy()
            _utils.check_unit_file(self._db)
            _utils.preprocess_unit_file(self._db)
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
    def allegiances_(self):
        """The list of allegiances."""
        return self.db_[["Allegiance", "allegiance_int"]].set_index("allegiance_int").drop_duplicates().squeeze()

    """################### HIDDEN ATTRIBUTE #################################"""

    @property
    def _segments(self):
        _seg_end = np.cumsum(self._unit_n)
        _seg_start = np.hstack((np.array([0], dtype=int), _seg_end[:-1]))
        return _seg_start, _seg_end

    @property
    def _teams(self) -> np.ndarray:
        self._is_instantiated()
        return np.asarray([self.db_.loc[u, "allegiance_int"] for u in self._unit_roster])

    """--------------------------------- PUBLIC FUNCTIONS ------------------------------------------------"""

    def create_army(self, army_set: List[Composite]):
        """
        Armies are groupings of (<'Unit Type'>, <number of units>). You can
        create one or more of these.

        We make use of the dataset (`db`) with army_set.

        We create the 'M' matrix, which is directly fed into any 'simulation' function.

        Parameters
        -------
        army_set : list of Composite
            A list of 'army groups' given as ('Unit Type', number of units)

        Returns self
        -------
        self
        """
        if not isinstance(army_set, (list, tuple)):
            raise TypeError("`army_set` must be a List/Tuple of Composites")

        if not all(isinstance(a, Composite) for a in army_set):
            raise TypeError("all instances within `army_set` must be composites.")

        self._comps = army_set
        # assign unit roster, n for roster
        self._unit_roster = [u.name.lower() for u in army_set]
        self._unit_n = [u.n for u in army_set]
        return self

    def apply_terrain(self, t: Optional[str] = None, res: float = .1):
        """
        Applies a Z-plane to the map that the Battle is occuring on by creating
        a bsm.Terrain object.

        Parameters
        -------
        t : str or bsm.Terrain
            Choose from [None, 'grid', 'contour']. Default is None. Contour looks
            the best. Decides how big/resolution to make the terra based on the
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

    def simulate(self, verbose: int = 0):
        """
        Runs the 'simulate_battle' algorithm. Creates and passes a copy to simulate..

        Returns np.ndarray of frames.
        """
        self._is_instantiated()
        # check for multiple teams
        if np.unique(self._teams).shape[0] <= 1:
            warnings.warn("Simulation halted - There is only one team present.", UserWarning)
            return self.sim_

        # set up M matrix from composition info
        self._presim()
        # re-generate terrain.
        self.T_.generate()
        # we cache a copy of the sim as well for convenience
        self._sim = sim_battle(np.copy(self.M_),
                               self.T_,
                               ret_frames=True)
        return self.sim_

    def simulate_k(self, k: int = 10):
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
            # pre-simulate fields.
            self._presim()
            # generate new terra
            self.T_.generate()

            for i in self._loading_bar(k):
                # run simulation
                team_counts = sim_battle(np.copy(self.M_),
                                         self.T_,
                                         ret_frames=False)
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
