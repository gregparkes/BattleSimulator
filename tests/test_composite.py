#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: gparkes
"""

import sys

sys.path.insert(0, "../")
import battlesim as bsm


def test_composite_init():
    # valid.
    bsm.Composite("example1", 50)
