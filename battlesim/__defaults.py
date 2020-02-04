#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:34:09 2019

@author: gparkes
"""
from pandas import DataFrame
from numpy import nan


__all__ = ["default_db"]


def default_db():
    """
    Returns a primary default database for bsm.Battle
    """
    return DataFrame({
        'Name': {0: 'Local Militia', 1: 'B1 battledroid', 2: 'Clone Trooper', 3: 'B2 battledroid',
                 4: 'ARC Trooper', 5: 'BX-series droid commando', 6: 'Clone Sharpshooter', 7: 'Battle droid assassin',
                 8: 'Clone Commando', 9: 'T-series Tactical Droid', 10: 'Magmaguard'},
        'Allegiance': {0: 'Republic', 1: 'CIS', 2: 'Republic', 3: 'CIS', 4: 'Republic', 5: 'CIS', 6: 'Republic',
                       7: 'CIS', 8: 'Republic', 9: 'CIS', 10: 'CIS'},
        'Type': {0: 'Standard', 1: 'Standard', 2: 'Standard', 3: 'Standard', 4: 'Elite', 5: 'Elite',
                 6:'Specialist', 7: 'Specialist', 8: 'Elite', 9: 'Elite', 10: 'Elite'},
        'Armor': {0: 15, 1: 35, 2: 50, 3: 100, 4: 100, 5: 70, 6: 50, 7: 60, 8: 100, 9: 200, 10: 400},
        'HP': {0: 20, 1: 20, 2: 35, 3: 40, 4: 100, 5: 50, 6: 40, 7: 20, 8: 120, 9: 50, 10: 200},
        'Damage': {0: 10, 1: 15, 2: 16, 3: 20, 4: 20, 5: 22, 6: 50, 7: 50, 8: 20, 9: 22, 10: 200},
        'Dmg Speed': {0: 1.0, 1: 1.0, 2: 1.2, 3: 1.2, 4: 0.9, 5: 1.3, 6: 0.2, 7: 0.15, 8: 1.5, 9: 1.5, 10: 0.15},
        'Range': {0: 2.0, 1: 2.5, 2: 3.0, 3: 3.5, 4: 5.0, 5: 7.0, 6: 15.0, 7: 15.0, 8: 3.0, 9: 3.0, 10: 0.4},
        'Movement Speed': {0: 0.6, 1: 0.5, 2: 0.7, 3: 0.4, 4: 1.0, 5: 0.8, 6: 0.2, 7: 0.2, 8: 0.9, 9: 0.7, 10: 0.4},
        'Accuracy': {0: 5, 1: 30, 2: 65, 3: 70, 4: 90, 5: 70, 6: 60, 7: 55, 8: 97, 9: 95, 10: 100},
        'Miss': {0: 30, 1: 35, 2: 75, 3: 40, 4: 95, 5: 95, 6: 40, 7: 45, 8: 97, 9: 90, 10: 40},
        'Shield': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 30, 9: 100, 10: 0},
        'Shield Regen': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.2, 9: 0.18, 10: nan}
    })
