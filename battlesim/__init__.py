# utf-8

"""
The initialization for the battlesim package.
"""

""" This block handles the import needs of the package """
hard_dependencies = ("numpy", "matplotlib", "pandas", "numba")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{0}: {1}".format(dependency, str(e)))

if missing_dependencies:
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(missing_dependencies))
del hard_dependencies, dependency, missing_dependencies
""" Block ends """


# imports
from ._battle import Battle
from .distrib import Composite, Sampling
from .terra import Terrain
from . import terra, distrib, plot, simulation
from . import __legacy

__version__ = "0.3.7"
__name__ = "battlesim"
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
