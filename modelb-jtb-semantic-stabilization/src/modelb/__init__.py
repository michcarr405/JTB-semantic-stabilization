"""
Model B (merged publication package)

This package merges:
- ModelB_verification_JRSI (baseline inheritance validation)
- Primitive-replication / ModelB Paper 3 suite (including lockdata workflows)

Core API
--------
- default_parameters: base parameter dictionary
- paper3_parameters / jrsi_parameters: presets for paper-specific runs
- run_replicate: run a single replicate simulation (alias for run_single_sim)
- sweep_inheritance: run inheritance sweeps (alias for run_inheritance_sweep)

Plotting
--------
See :mod:`modelb.viz.plotting` for the Primitive-replication color scheme and plotting helpers.
"""
from .config import default_parameters
from .paper_configs import paper3_parameters, jrsi_parameters
from .simulation import run_single_sim as run_replicate
from .sweeps import run_inheritance_sweep as sweep_inheritance

__all__ = [
    "default_parameters",
    "paper3_parameters",
    "jrsi_parameters",
    "run_replicate",
    "sweep_inheritance",
]
