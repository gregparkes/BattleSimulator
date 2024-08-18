"""The initialization for the battlesim package."""

# imports
from . import __legacy  # noqa: F401
from . import distrib, plot, simulation, terra  # noqa: F401
from ._battle import Battle  # noqa: F401
from .distrib import Composite, Sampling  # noqa: F401
from .terra import Terrain  # noqa: F401

__version__ = "0.3.7"
__name__ = "battlesim"  # noqa: W0622
__doc__ = """
battlesim - Modelling and animating simulated battles between units in Python.
==============================================================================

**battlesim** is a Python package providing TABS (totally-accurate-battle-simulator)-
style combat designed to entertain, inform and acts as a platform for simulation and
modelling within a games-design context. Simulations are designed not only to be
comprehensive and flexible, but also fast by relying on just-in-time compiling.

Main Features
-------------
Here are just a few things that battlesim aims to do well:

    - Formulate your simulation in a few lines of code from scratch.
    - Scales up to thousands (and 10s of thousands) of units
    - Flexibility: unit values are taken from a data file with flexible AI options
    - Performance: Just-in-time compiling (JIT) can manage thousands of units
    - Visualisation: Animations can be customized to change look-and-feel
"""
