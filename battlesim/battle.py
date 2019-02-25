#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gparkes
"""
from . import utils
import pandas as pd

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

    def __init__(self, score_file):
        """
        Initialise a Battle object, and pass in the relevant scoring file.
        """
        # set the filepath. do nothing else.
        self.fpath = score_file
        self.db = None


    def __enter__(self):
        return self.init()


    def __exit__(self, type, value, traceback):
        # do nothing
        pass


    def init(self):
        """
        Initialise the Battle object using the filepath provided.
        """
        self.db = utils.import_and_check_unit_file(self.fpath)
        # set name to index
        self.db.set_index("Name", inplace=True)
        self.db["allegiance_int"] = pd.factorize(self.db.Allegiance)[0]
        return self
