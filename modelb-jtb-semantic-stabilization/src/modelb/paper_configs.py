"""
Paper-specific parameter presets.

We keep a single canonical default parameter set in :func:`modelb.config.default_parameters`.
This module provides small, explicit overrides for particular manuscripts / suites.
"""
from __future__ import annotations

from typing import Dict, Any
from .config import default_parameters

def paper3_parameters() -> Dict[str, Any]:
    """
    Parameters used by the Paper 3 suite in the Primitive-replication package.

    Notes
    -----
    The original Paper 3 package used shorter sweeps by default
    (n_gens=15, n_reps=2) and then varied settings within scripts.
    """
    p = default_parameters()
    # Keep Paper 3 defaults explicit for reproducibility.
    p["n_gens"] = 15
    p["n_reps"] = 2
    return p

def jrsi_parameters() -> Dict[str, Any]:
    """
    Parameters used by the JRSI baseline inheritance validation suite.

    Notes
    -----
    The original JRSI validation used longer sweeps by default
    (n_gens=150, n_reps=20).
    """
    p = default_parameters()
    p["n_gens"] = 150
    p["n_reps"] = 20
    return p
