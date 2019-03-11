#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gparkes
"""
import numpy as np
import pandas as pd

from .army import Army, DelayArmy
from . import ai
from . import utils


class Battle(object):
    """
    The Battle object provides a context for a single or multiple battles in a
    row using simular parameters.

    For instance, a user may do something like:

    with Battle('name of unit scoring file') as b:
        # create armies
        # run simulation

    # plot animation
    """

    def __init__(self, score_file, delayed=True):
        """
        Initialise a Battle object, and pass in the relevant scoring file.

        Parameters
        -------
        score_file : str
            A link to a unit-scores.csv file.
        delayed : bool
            If True, objects passed in Battle.add() are given data.
        """
        # set the filepath. do nothing else.
        self.fpath = score_file
        self._load()
        self._delayed = delayed
        self._unit_names = set(self._db.index)
        self._unit_to_idx = dict(zip(self._db.index, range(self._db.shape[0])))
        self._armies = []
        self._instantiated = False


    def __enter__(self):
        return self._load()


    def __exit__(self, type, value, traceback):
        # do nothing
        pass

    ########################### PROPERTIES #######################

    def _get_armies(self):
        return self._armies

    def _get_instantiation(self):
        return self._instantiated

    units_ = property(_get_armies)
    instance_ = property(_get_instantiation)

    def _load(self):
        """
        Initialise the Battle object using the filepath provided.
        """
        self._db = utils.import_and_check_unit_file(self.fpath)
        # set name to index
        self._db.set_index("Name", inplace=True)
        self._db["allegiance_int"] = pd.factorize(self._db.Allegiance)[0]
        return self


    def add(self, armies):
        """
        Add DelayArmy objects to the roster of the fight.

        Parameters
        -------
        armies : list of Unit, Army or DelayArmy
            The groups of objects used to define the fight group.

        Returns
        -------
        self
        """
        if isinstance(armies, (DelayArmy)):
            # convert to a list
            armies = [armies]
        elif isinstance(armies, (list, tuple)):
            pass
        else:
            raise ValueError("armies is type '{}', must be list or tuple".format(type(armies)))
        # extend the armies list
        # check that every element in the list is a Unit, Army or DelayArmy
        assert sum([isinstance(a, DelayArmy) for a in armies]) == len(armies), "Err: not all elements are 'DelayArmy' type"
        self._armies.extend(armies)
        return self


    def instantiate(self):
        """
        Instantiates the 'Battle' object given the Armies and Units assigned to it,
        using the linked unit-scores. Should create Matrix objects to use in any
        resulting 'simulation' function.

        We create two 'changeable' matrices with the following columns:
            (float)
            0 - HP
            1 - X
            2 - Y
            3 - dX
            4 - dY

            (int)
            0 - Target index

        We also create two 'static' matrices with the following columns:
            (float)
            0 - Move speed
            1 - Accuracy
            2 - Damage
            3 - Dodge
            4 - Range

            (integers)
            0 - Allegiance
            1 - AI type
        """
        N = sum([a.N_ for a in self._armies])
        # create an 'index range' for each army.
        cum_N = np.cumsum(np.array([a.N_ for a in self._armies]))
        ranges = list(zip(cum_N-self._armies[0].N_, cum_N))
        for i, r in enumerate(ranges):
            self._armies[i].index_range_ = r

        st_fdict = dict(zip(["Movement Speed","Accuracy","Damage","Miss","Range"], range(5)))
        ai_dict = dict(zip(["random", "nearest"], range(2)))

        self.ch_f = np.zeros((N,5))
        self.ch_i = np.zeros((N,), dtype=np.int)
        self.st_f = np.zeros((N,5))
        self.st_i = np.zeros((N,2), dtype=np.int)

        # fill based on armies
        # HP
        self.ch_f[:,0] = np.hstack([
            np.repeat(self._db.loc[a.name_].HP, a.N_) for a in self.units_
        ])
        # use an algorithm to assign X and Y for each army, given it's position params
        self.ch_f[:,1:3] = np.vstack([utils.positions_from_spec(a.pos_, a.N_)
            for a in self.units_])

        # static matrices
        for col, ids in st_fdict.items():
            self.st_f[:,ids] = np.hstack([
                np.repeat(self._db.loc[a.name_][col], a.N_) for a in self.units_
            ])

        self.st_i[:,0] = np.hstack([
            np.repeat(self._db.loc[a.name_].allegiance_int, a.N_) for a in self.units_
        ])
        self.st_i[:,1] = np.hstack([
            np.repeat(ai_dict[a.ai_],a.N_) for a in self.units_
        ])

        # fetch a healthy target for each unit using allegiance positions
        self.ch_i = np.hstack([ai.init_ai_random2(a,self.st_i[:,0])
            for a in self.units_])

        # assign dX and dY direction based on a chosen target.
        self._instantiated = True

        return NotImplemented


    def __repr__(self):
        return "Battle(n='%d')" % (sum([a.N_ for a in self._armies]))

