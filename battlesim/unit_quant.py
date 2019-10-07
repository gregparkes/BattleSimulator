#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 15:40:34 2019

@author: gparkes

This class is for quantifying the usefulness of each unit.

This relies on the default columns being made available to it.
"""

import pandas as pd
import numpy as np

from . import utils

def rank_score(db, w=None):
    """
    Using the dataset, we can 'rank score' each unit according to it's relation
    to every other unit.

    We can do this uniformly over given features, or using weighting.

    We then sum the normalized ranks and take the average for a 'score' for each
    unit.

    Parameters
    -------
    db : pd.DataFrame
        The dataset in question.
    w : np.ndarray (4, ) [0..1]
        Weights for each rankable column. Currently six.
        [Damage, Range, Movement, HP]
    Returns
    -------
    p : pd.Series
        The scores for each unit.
    """
    if w is None:
        w = np.ones(4,)
    else:
        assert len(w) == 4, "len(w) must equal 6"

    rankable_columns = ["Damage","Range","Movement Speed", "HP"]
    # check they are there!
    utils.check_in_list(db.columns.tolist(), rankable_columns)

    Rank = pd.DataFrame(
        [utils.max_norm(db[r].rank()) * weight for weight, r in zip(w, rankable_columns)]
    )
    return Rank.mean().sort_values()
