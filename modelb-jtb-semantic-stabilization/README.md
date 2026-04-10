# Model B JTB semantic stabilization package

This repository contains the code, analysis workflow, lockdata, and figure-generation pipeline used for the Journal of Theoretical Biology manuscript on semantic stabilization in Model B under noisy compositional transmission.

## Contents

- `src/modelb/` — core Model B simulation and mutual-information code
- `scripts/jtb_semantic_stabilization_suite.py` — end-to-end JTB reproduction script
- `scripts/paper3_overlays.py` — overlay definitions used by the JTB suite
- `outputs/jtb_semantic_stabilization/data/` — lockdata tables and statistical outputs used for analysis
- `outputs/jtb_semantic_stabilization/figs/` — manuscript figures in PNG and PDF format
- `tests/` — minimal smoke test

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Optional speed dependency:

```bash
python -m pip install -e ".[speed]"
```

## Reproducing the JTB analysis

### Plot only from bundled lockdata

```bash
python scripts/jtb_semantic_stabilization_suite.py --plot
```

### Full rerun plus figure regeneration

```bash
python scripts/jtb_semantic_stabilization_suite.py --run --plot
```

### Full rerun overwriting existing lockdata

```bash
python scripts/jtb_semantic_stabilization_suite.py --run --plot --force
```

## Outputs

The JTB suite writes results to:

- `outputs/jtb_semantic_stabilization/data/`
- `outputs/jtb_semantic_stabilization/figs/`

Key analysis files include:

- `overlay_timeseries.csv`
- `overlay_summary.csv`
- `overlay_stat_tests.json`
- `sweep_timeseries.csv`
- `sweep_summary.csv`
- `sweep_point_estimates.csv`
- `sweep_pairwise_tests.csv`
- `sweep_stat_tests.json`
- `lockdata_metadata.json`


