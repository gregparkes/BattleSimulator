"""Internal simulation methods."""

import numpy as np

from . import _ai, _damage, _hit, _move, _target  # noqa: F401
from ._simulator_fast import simulate_battle as _simulate_battle  # noqa: F401


def simulate_battle(M, *args, **kwargs):
    """
    Run a battle simulation and return a result that always covers every team
    present in the unit database loaded into the battle object.

    If the unit matrix ``M`` does not contain all teams from the database,
    missing teams are represented as zero-health teams so downstream code that
    assumes one result column per database team does not fail.
    """
    F = _simulate_battle(M, *args, **kwargs)
    if hasattr(M, "db_") and F.ndim >= 2:
        n_teams_db = len(np.unique(M.db_["Allegiance"]))
        if F.shape[-1] < n_teams_db:
            F = np.pad(F, ((0, 0), (0, n_teams_db - F.shape[-1])), mode="constant", constant_values=0.0)
    return F
