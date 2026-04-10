#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JTB semantic stabilization suite for Model B.

Purpose
-------
Create a publication-ready, lockdata-based package for a Journal of Theoretical
Biology manuscript focused on replication as a semantic stabilization mechanism
under noisy compositional transmission.

What this script does
---------------------
1) Re-runs the overlay comparison suite at a focal inheritance fidelity
   (default pinherit=0.8) and writes lockdata.
2) Runs an inheritance-fidelity sweep for baseline and representative strong
   overlay conditions (default: baseline, TAC_s1.0, PB_c1.0, AR_s1.0).
3) Computes statistical tests appropriate to the sweep:
   - Fisher exact tests for stabilization probability versus baseline at the
     same inheritance fidelity.
   - Mann-Whitney U tests for continuous metrics versus baseline at the same
     inheritance fidelity.
   - Benjamini-Hochberg FDR correction within each metric family.
   - Bootstrap confidence intervals for means and Wilson intervals for
     stabilization probabilities.
4) Plots all figures from saved lockdata without rerunning the simulator.

Usage
-----
# Install once from repo root
python -m pip install -e .

# Full run + plots
python scripts/jtb_semantic_stabilization_suite.py --run --plot

# Plot only from existing lockdata
python scripts/jtb_semantic_stabilization_suite.py --plot

# Overwrite existing lockdata
python scripts/jtb_semantic_stabilization_suite.py --run --force

Environment overrides
---------------------
JTB_N_GENS            default 200
JTB_N_REPS            default 30
JTB_N_CALIB           default min(10, max(3, n_reps//2))
JTB_STAB_WINDOW       default 20
JTB_FOCAL_INHERIT     default 0.8
JTB_SWEEP_INHERITS    comma-separated list, default 0.50,0.60,0.70,0.80,0.90,1.00
JTB_TAC_ERROR         default 0.01
JTB_AR_ERROR          default 0.0

Outputs
-------
outputs/jtb_semantic_stabilization/
  data/
    overlay_timeseries.csv
    overlay_summary.csv
    overlay_stat_tests.json
    sweep_timeseries.csv
    sweep_summary.csv
    sweep_point_estimates.csv
    sweep_pairwise_tests.csv
    sweep_stat_tests.json
    lockdata_metadata.json
  figs/
    overlay_fitness_timecourse.png/.pdf
    overlay_mi_timecourse.png/.pdf
    overlay_dwell_bars.png/.pdf
    overlay_p_stab_bars.png/.pdf
    overlay_fitness_final_bars.png/.pdf
    overlay_lag_hist.png/.pdf
    sweep_p_stab_vs_inherit.png/.pdf
    sweep_dwell_vs_inherit.png/.pdf
    sweep_delta_fitness_vs_inherit.png/.pdf
    sweep_delta_mi_vs_inherit.png/.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ensure local imports work when run from anywhere
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

os.environ.setdefault("PYTHONHASHSEED", "0")

from modelb.config import default_parameters
from modelb.core import compute_fitnesses_and_observations
from modelb.mi import compute_mutual_information
from paper3_overlays import OverlayConfig, apply_overlay_to_reproduction

OUTDIR = REPO_ROOT / "outputs" / "jtb_semantic_stabilization"
DATADIR = OUTDIR / "data"
FIGDIR = OUTDIR / "figs"


@dataclass(frozen=True)
class Condition:
    name: str
    inherit_prob: float
    overlay: OverlayConfig


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def make_seed(condition_name: str, rep: int, seed_base: int = 2_000_000) -> int:
    return int(seed_base + stable_int(condition_name) + rep)


def parse_float_list(env_name: str, default: Sequence[float]) -> List[float]:
    raw = os.environ.get(env_name)
    if not raw:
        return list(default)
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def _tol_colors(n: int) -> List[str]:
    base = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
    ]
    return [base[i % len(base)] for i in range(n)]


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    })


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def savefig(path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(path_no_ext) + ".png", dpi=300)
    plt.savefig(str(path_no_ext) + ".pdf")
    plt.close()


def bootstrap_ci(x: np.ndarray, fn=np.mean, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    n = x.size
    for i in range(n_boot):
        samp = x[rng.integers(0, n, size=n)]
        boots[i] = float(fn(samp))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bh_adjust(p_values: Iterable[float]) -> List[float]:
    vals = np.asarray(list(p_values), dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(vals)
    if not np.any(ok):
        return out.tolist()
    idx = np.where(ok)[0]
    pv = vals[idx]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = np.empty_like(ranked)
    prev = 1.0
    m = len(ranked)
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = prev
    out_idx = idx[order]
    out[out_idx] = np.clip(adj, 0.0, 1.0)
    return out.tolist()


def mannwhitney(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    y = np.asarray(y)
    try:
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        return {"U": float(stat), "p": float(p)}
    except Exception:
        return {"U": float("nan"), "p": float("nan")}


def fisher_exact_from_bool(a: np.ndarray, b: np.ndarray) -> Dict[str, object]:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    table = np.array([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]], dtype=int)
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact(table, alternative="two-sided")
        return {"p": float(p), "table": table.tolist()}
    except Exception:
        return {"p": float("nan"), "table": table.tolist()}


def first_crossing_time(arr: np.ndarray, thr: float) -> Optional[int]:
    idx = np.where(np.asarray(arr) >= thr)[0]
    return int(idx[0]) if idx.size else None


def stabilization_time(arr: np.ndarray, thr: float, window: int) -> Optional[int]:
    arr = np.asarray(arr)
    if arr.size < window:
        return None
    ok = arr >= thr
    consec = 0
    for t in range(arr.size):
        consec = consec + 1 if ok[t] else 0
        if consec >= window:
            return int(t - window + 1)
    return None


def dwell_fraction(arr: np.ndarray, thr: float) -> float:
    return float(np.mean(np.asarray(arr) >= thr))


def n_upcrossings(arr: np.ndarray, thr: float) -> int:
    arr = np.asarray(arr)
    above = arr >= thr
    if arr.size < 2:
        return 0
    return int(np.sum((~above[:-1]) & above[1:]))


def init_population(n_cells: int, n_seqs: int, seq_len: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 4, size=(n_cells, n_seqs, seq_len), dtype=np.int8)


def run_single_sim_with_overlay(
    *,
    seed: int,
    params: dict,
    inherit_prob: float,
    overlay: OverlayConfig,
    motif_affinity_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = init_population(params["n_cells"], params["n_seqs"], params["seq_len"], rng)
    fit_hist: List[float] = []
    mi_hist: List[float] = []
    for _ in range(int(params["n_gens"])):
        fitnesses, motifs, mets = compute_fitnesses_and_observations(
            pop,
            motif_affinity_matrix,
            params["segment_favored_met"],
            params["bias_strength"],
            params["productive_pairs"],
            params["anti_pairs"],
            params["reward_strength"],
            params["penalty_strength"],
            params["temperature"],
        )
        fit_hist.append(float(np.mean(fitnesses)))
        mi_hist.append(float(compute_mutual_information(mets, motifs, params["n_metabolites"])))
        pop = apply_overlay_to_reproduction(
            pop=pop,
            fitnesses=fitnesses,
            mu=params["mutation_rate"],
            inherit_prob=inherit_prob,
            overlay=overlay,
            rng=rng,
        )
    return np.asarray(fit_hist, dtype=float), np.asarray(mi_hist, dtype=float)


def default_overlay_conditions(focal_inherit: float, tac_error: float, ar_error: float) -> List[Condition]:
    conds: List[Condition] = [Condition("baseline", focal_inherit, OverlayConfig(kind="none"))]
    for s in [0.25, 0.5, 0.75, 1.0]:
        conds.append(Condition(f"TAC_s{s}", focal_inherit, OverlayConfig(kind="TAC", strength=s, error=tac_error)))
    for c in [0.5, 1.0]:
        conds.append(Condition(f"PB_c{c}", focal_inherit, OverlayConfig(kind="PB", corr=c)))
    for s in [0.5, 1.0]:
        conds.append(Condition(f"AR_s{s}", focal_inherit, OverlayConfig(kind="AR", strength=s, error=ar_error)))
    return conds


def default_sweep_conditions(inherit_probs: Sequence[float], tac_error: float, ar_error: float) -> List[Condition]:
    conds: List[Condition] = []
    for p in inherit_probs:
        conds.extend([
            Condition(f"baseline__p{p:.2f}", p, OverlayConfig(kind="none")),
            Condition(f"TAC_s1.0__p{p:.2f}", p, OverlayConfig(kind="TAC", strength=1.0, error=tac_error)),
            Condition(f"PB_c1.0__p{p:.2f}", p, OverlayConfig(kind="PB", corr=1.0)),
            Condition(f"AR_s1.0__p{p:.2f}", p, OverlayConfig(kind="AR", strength=1.0, error=ar_error)),
        ])
    return conds


def summarize_replicate(
    fit: np.ndarray,
    mi: np.ndarray,
    mi_thr: float,
    window: int,
    condition: Condition,
    rep: int,
    seed: int,
) -> Dict[str, object]:
    t_cross = first_crossing_time(mi, mi_thr)
    t_stab = stabilization_time(mi, mi_thr, window)
    dwell = dwell_fraction(mi, mi_thr)
    n_innov = n_upcrossings(mi, mi_thr)
    pre = max(5, int(len(fit)) // 10)
    fit0 = float(np.mean(fit[:pre]))
    fit_delta = max(float(np.std(fit[:pre])), 0.01)
    t_fit = stabilization_time(fit, fit0 + fit_delta, window)
    lag = int(t_fit - t_stab) if (t_fit is not None and t_stab is not None) else None
    return {
        "condition": condition.name,
        "rep": rep,
        "seed": seed,
        "inherit_prob": float(condition.inherit_prob),
        "overlay_kind": condition.overlay.kind,
        "overlay_strength": float(condition.overlay.strength),
        "overlay_error": float(condition.overlay.error),
        "overlay_corr": float(condition.overlay.corr),
        "mi_thr": float(mi_thr),
        "stab_window": int(window),
        "t_cross": t_cross,
        "t_stab": t_stab,
        "dwell": float(dwell),
        "n_innov": int(n_innov),
        "fitness_final": float(fit[-1]),
        "mi_final": float(mi[-1]),
        "delta_fitness": float(fit[-1] - fit[0]),
        "delta_mi": float(mi[-1] - mi[0]),
        "t_fit": t_fit,
        "lag_fit_minus_stab": lag,
    }


def calibrate_mi_threshold(
    params: dict,
    motif_affinity_matrix: np.ndarray,
    inherit_prob: float,
    n_calib: int,
) -> float:
    mi_vals: List[float] = []
    for r in range(n_calib):
        _, mi = run_single_sim_with_overlay(
            seed=9_000_000 + r,
            params=params,
            inherit_prob=inherit_prob,
            overlay=OverlayConfig(kind="none"),
            motif_affinity_matrix=motif_affinity_matrix,
        )
        mi_vals.extend(mi.tolist())
    return float(np.quantile(np.asarray(mi_vals, dtype=float), 0.90))


def run_condition_grid(
    conditions: Sequence[Condition],
    params: dict,
    motif_affinity_matrix: np.ndarray,
    mi_thr: float,
    window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts_rows: List[Dict[str, object]] = []
    sum_rows: List[Dict[str, object]] = []
    n_reps = int(params["n_reps"])
    for cond in conditions:
        print(f"[JTB] Running {cond.name} ({n_reps} reps, inherit={cond.inherit_prob:.2f})")
        for rep in range(n_reps):
            seed = make_seed(cond.name, rep)
            fit, mi = run_single_sim_with_overlay(
                seed=seed,
                params=params,
                inherit_prob=cond.inherit_prob,
                overlay=cond.overlay,
                motif_affinity_matrix=motif_affinity_matrix,
            )
            for t in range(int(params["n_gens"])):
                ts_rows.append({
                    "condition": cond.name,
                    "rep": rep,
                    "seed": seed,
                    "t": t,
                    "fitness": float(fit[t]),
                    "mi": float(mi[t]),
                    "inherit_prob": float(cond.inherit_prob),
                    "overlay_kind": cond.overlay.kind,
                    "overlay_strength": float(cond.overlay.strength),
                    "overlay_error": float(cond.overlay.error),
                    "overlay_corr": float(cond.overlay.corr),
                })
            sum_rows.append(summarize_replicate(fit, mi, mi_thr, window, cond, rep, seed))
    return pd.DataFrame(ts_rows), pd.DataFrame(sum_rows)


def compute_overlay_stats(df_sum: pd.DataFrame, conditions: Sequence[Condition]) -> Dict[str, object]:
    baseline = df_sum[df_sum["condition"] == "baseline"].copy()
    base_stab = baseline["t_stab"].notna().to_numpy()
    base_dwell = baseline["dwell"].to_numpy()
    base_fit = baseline["fitness_final"].to_numpy()
    base_n_innov = baseline["n_innov"].to_numpy()
    base_dfit = baseline["delta_fitness"].to_numpy()
    base_dmi = baseline["delta_mi"].to_numpy()
    out: Dict[str, object] = {"comparisons": {}}
    for cond in conditions:
        if cond.name == "baseline":
            continue
        sub = df_sum[df_sum["condition"] == cond.name].copy()
        sub_stab = sub["t_stab"].notna().to_numpy()
        sub_dwell = sub["dwell"].to_numpy()
        sub_fit = sub["fitness_final"].to_numpy()
        sub_n_innov = sub["n_innov"].to_numpy()
        sub_dfit = sub["delta_fitness"].to_numpy()
        sub_dmi = sub["delta_mi"].to_numpy()
        out["comparisons"][cond.name] = {
            "stabilization_fisher": fisher_exact_from_bool(base_stab, sub_stab),
            "dwell_mannwhitney": mannwhitney(base_dwell, sub_dwell),
            "fitness_final_mannwhitney": mannwhitney(base_fit, sub_fit),
            "innovation_mannwhitney": mannwhitney(base_n_innov, sub_n_innov),
            "delta_fitness_mannwhitney": mannwhitney(base_dfit, sub_dfit),
            "delta_mi_mannwhitney": mannwhitney(base_dmi, sub_dmi),
            "effect_sizes": {
                "dwell_delta_mean": float(np.mean(sub_dwell) - np.mean(base_dwell)),
                "fitness_final_delta_mean": float(np.mean(sub_fit) - np.mean(base_fit)),
                "delta_fitness_delta_mean": float(np.mean(sub_dfit) - np.mean(base_dfit)),
                "delta_mi_delta_mean": float(np.mean(sub_dmi) - np.mean(base_dmi)),
                "innovation_delta_mean": float(np.mean(sub_n_innov) - np.mean(base_n_innov)),
                "p_stab_baseline": float(np.mean(base_stab)),
                "p_stab_cond": float(np.mean(sub_stab)),
            },
        }
    return out


def compute_sweep_stats(df_sum: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    families = ["p_stab", "dwell", "delta_fitness", "delta_mi", "fitness_final"]
    point_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    grouped = df_sum.groupby(["overlay_kind", "inherit_prob"], sort=True)
    for (overlay_kind, inherit_prob), sub in grouped:
        stab = sub["t_stab"].notna().to_numpy()
        k = int(stab.sum())
        n = int(stab.size)
        p_stab = float(k / n)
        p_lo, p_hi = wilson_interval(k, n)
        dfit = sub["delta_fitness"].to_numpy()
        dmi = sub["delta_mi"].to_numpy()
        dwell = sub["dwell"].to_numpy()
        fit_final = sub["fitness_final"].to_numpy()
        point_rows.append({
            "overlay_kind": overlay_kind,
            "inherit_prob": float(inherit_prob),
            "n_reps": n,
            "n_stab": k,
            "p_stab": p_stab,
            "p_stab_ci_lo": p_lo,
            "p_stab_ci_hi": p_hi,
            "mean_dwell": float(np.mean(dwell)),
            "mean_dwell_ci_lo": bootstrap_ci(dwell, np.mean, seed=1)[0],
            "mean_dwell_ci_hi": bootstrap_ci(dwell, np.mean, seed=1)[1],
            "mean_delta_fitness": float(np.mean(dfit)),
            "mean_delta_fitness_ci_lo": bootstrap_ci(dfit, np.mean, seed=2)[0],
            "mean_delta_fitness_ci_hi": bootstrap_ci(dfit, np.mean, seed=2)[1],
            "mean_delta_mi": float(np.mean(dmi)),
            "mean_delta_mi_ci_lo": bootstrap_ci(dmi, np.mean, seed=3)[0],
            "mean_delta_mi_ci_hi": bootstrap_ci(dmi, np.mean, seed=3)[1],
            "mean_fitness_final": float(np.mean(fit_final)),
            "mean_fitness_final_ci_lo": bootstrap_ci(fit_final, np.mean, seed=4)[0],
            "mean_fitness_final_ci_hi": bootstrap_ci(fit_final, np.mean, seed=4)[1],
        })

    inherit_probs = sorted(df_sum["inherit_prob"].unique().tolist())
    overlay_kinds = sorted([k for k in df_sum["overlay_kind"].unique().tolist() if k != "none"])

    for p in inherit_probs:
        base = df_sum[(df_sum["inherit_prob"] == p) & (df_sum["overlay_kind"] == "none")].copy()
        base_stab = base["t_stab"].notna().to_numpy()
        base_dwell = base["dwell"].to_numpy()
        base_dfit = base["delta_fitness"].to_numpy()
        base_dmi = base["delta_mi"].to_numpy()
        base_fit = base["fitness_final"].to_numpy()
        for kind in overlay_kinds:
            sub = df_sum[(df_sum["inherit_prob"] == p) & (df_sum["overlay_kind"] == kind)].copy()
            pair_rows.append({
                "inherit_prob": float(p),
                "overlay_kind": kind,
                "metric": "p_stab",
                "raw_p": float(fisher_exact_from_bool(base_stab, sub["t_stab"].notna().to_numpy())["p"]),
                "effect": float(np.mean(sub["t_stab"].notna().to_numpy()) - np.mean(base_stab)),
            })
            for metric, bvals, svals in [
                ("dwell", base_dwell, sub["dwell"].to_numpy()),
                ("delta_fitness", base_dfit, sub["delta_fitness"].to_numpy()),
                ("delta_mi", base_dmi, sub["delta_mi"].to_numpy()),
                ("fitness_final", base_fit, sub["fitness_final"].to_numpy()),
            ]:
                mw = mannwhitney(bvals, svals)
                pair_rows.append({
                    "inherit_prob": float(p),
                    "overlay_kind": kind,
                    "metric": metric,
                    "raw_p": float(mw["p"]),
                    "effect": float(np.mean(svals) - np.mean(bvals)),
                    "U": float(mw["U"]),
                })

    pair_df = pd.DataFrame(pair_rows)
    pair_df["p_fdr"] = np.nan
    for metric in families:
        mask = pair_df["metric"] == metric
        pair_df.loc[mask, "p_fdr"] = bh_adjust(pair_df.loc[mask, "raw_p"].tolist())

    summary = {
        "familywise_metrics": families,
        "n_pairwise_tests": int(len(pair_df)),
        "inherit_probs": inherit_probs,
        "overlay_kinds": overlay_kinds,
    }
    return pd.DataFrame(point_rows), pair_df, summary


def sort_overlay_name(name: str) -> Tuple[int, float]:
    if name == "baseline":
        return (0, 0.0)
    if name.startswith("TAC_s"):
        return (1, float(name.split("TAC_s")[1]))
    if name.startswith("PB_c"):
        return (2, float(name.split("PB_c")[1]))
    if name.startswith("AR_s"):
        return (3, float(name.split("AR_s")[1]))
    return (9, 0.0)


def overlay_label(kind: str) -> str:
    return {"none": "Baseline", "TAC": "TAC s=1.0", "PB": "PB c=1.0", "AR": "AR s=1.0"}.get(kind, kind)


def plot_timecourse(df_ts: pd.DataFrame, y: str, out: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    conds = sorted(df_ts["condition"].unique().tolist(), key=sort_overlay_name)
    colors = _tol_colors(len(conds))
    for i, cond in enumerate(conds):
        sub = df_ts[df_ts["condition"] == cond]
        stats = sub.groupby("t")[y].agg(["mean", "sem"]).reset_index()
        ax.plot(stats["t"], stats["mean"], lw=1.2, color=colors[i], label=cond)
        ax.fill_between(stats["t"], stats["mean"] - stats["sem"], stats["mean"] + stats["sem"], color=colors[i], alpha=0.18)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean fitness" if y == "fitness" else "Mutual information")
    _clean_axes(ax)
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(out)


def plot_metric_bars(df_sum: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    set_style()
    conds = sorted(df_sum["condition"].unique().tolist(), key=sort_overlay_name)
    colors = _tol_colors(len(conds))
    means, sems = [], []
    for cond in conds:
        vals = df_sum.loc[df_sum["condition"] == cond, metric].to_numpy(dtype=float)
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=sems, color=colors, edgecolor="none", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    _clean_axes(ax)
    savefig(out)


def plot_pstab_bars(df_sum: pd.DataFrame, out: Path) -> None:
    set_style()
    conds = sorted(df_sum["condition"].unique().tolist(), key=sort_overlay_name)
    colors = _tol_colors(len(conds))
    means, los, his = [], [], []
    for cond in conds:
        stab = df_sum.loc[df_sum["condition"] == cond, "t_stab"].notna().to_numpy()
        k, n = int(stab.sum()), int(stab.size)
        p = k / n
        lo, hi = wilson_interval(k, n)
        means.append(p)
        los.append(p - lo)
        his.append(hi - p)
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(conds))
    ax.bar(x, means, yerr=np.vstack([los, his]), color=colors, edgecolor="none", capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right")
    ax.set_ylabel("Stabilization probability")
    ax.set_ylim(0.0, 1.05)
    _clean_axes(ax)
    savefig(out)


def plot_lag_hist(df_sum: pd.DataFrame, out: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    conds = sorted(df_sum["condition"].unique().tolist(), key=sort_overlay_name)
    colors = _tol_colors(len(conds))
    handles: List[Patch] = []
    for i, cond in enumerate(conds):
        vals = df_sum.loc[(df_sum["condition"] == cond) & df_sum["lag_fit_minus_stab"].notna(), "lag_fit_minus_stab"].to_numpy(dtype=float)
        total = int((df_sum["condition"] == cond).sum())
        n = len(vals)
        label = f"{cond} (n={n}/{total})"
        if n > 0:
            ax.hist(vals, bins=20, alpha=0.45, color=colors[i])
            handles.append(Patch(facecolor=colors[i], edgecolor="none", alpha=0.45, label=label))
        else:
            handles.append(Patch(facecolor=colors[i], edgecolor="none", alpha=0.18, label=label))
    ax.set_xlabel(r"Lag = $T_{fit} - T_{stab}$")
    ax.set_ylabel("Count")
    _clean_axes(ax)
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(out)


def plot_sweep_lines(point_df: pd.DataFrame, metric: str, ylabel: str, out: Path, ci_lo: str, ci_hi: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    kinds = ["none", "TAC", "PB", "AR"]
    colors = _tol_colors(len(kinds))
    for i, kind in enumerate(kinds):
        sub = point_df[point_df["overlay_kind"] == kind].sort_values("inherit_prob")
        ax.plot(sub["inherit_prob"], sub[metric], marker="o", lw=1.2, ms=3.5, color=colors[i], label=overlay_label(kind))
        ax.fill_between(sub["inherit_prob"], sub[ci_lo], sub[ci_hi], color=colors[i], alpha=0.18)
    ax.set_xlabel("Inheritance probability")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    _clean_axes(ax)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    savefig(out)


def run_all(datadir: Path = DATADIR, figdir: Path = FIGDIR) -> None:
    datadir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    params = default_parameters()
    params["n_gens"] = int(os.environ.get("JTB_N_GENS", "200"))
    params["n_reps"] = int(os.environ.get("JTB_N_REPS", "30"))
    focal_inherit = float(os.environ.get("JTB_FOCAL_INHERIT", "0.8"))
    inherit_probs = parse_float_list("JTB_SWEEP_INHERITS", [0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    tac_error = float(os.environ.get("JTB_TAC_ERROR", "0.01"))
    ar_error = float(os.environ.get("JTB_AR_ERROR", "0.0"))
    n_calib = int(os.environ.get("JTB_N_CALIB", str(min(10, max(3, params["n_reps"] // 2)))))
    window = int(os.environ.get("JTB_STAB_WINDOW", "20"))
    motif_affinity_matrix = params["motif_affinity_matrix"]

    print(f"[JTB] Calibrating MI* at focal inheritance {focal_inherit:.2f} using {n_calib} baseline replicates")
    mi_thr = calibrate_mi_threshold(params, motif_affinity_matrix, focal_inherit, n_calib)
    print(f"[JTB] MI* = {mi_thr:.6f}; window = {window}")

    overlay_conditions = default_overlay_conditions(focal_inherit, tac_error, ar_error)
    df_ts_overlay, df_sum_overlay = run_condition_grid(overlay_conditions, params, motif_affinity_matrix, mi_thr, window)
    df_ts_overlay.to_csv(datadir / "overlay_timeseries.csv", index=False)
    df_sum_overlay.to_csv(datadir / "overlay_summary.csv", index=False)
    overlay_stats = compute_overlay_stats(df_sum_overlay, overlay_conditions)
    overlay_stats.update({
        "mi_threshold": float(mi_thr),
        "focal_inherit": float(focal_inherit),
        "n_reps": int(params["n_reps"]),
        "n_gens": int(params["n_gens"]),
        "stab_window": int(window),
    })
    with open(datadir / "overlay_stat_tests.json", "w") as f:
        json.dump(overlay_stats, f, indent=2)

    sweep_conditions = default_sweep_conditions(inherit_probs, tac_error, ar_error)
    df_ts_sweep, df_sum_sweep = run_condition_grid(sweep_conditions, params, motif_affinity_matrix, mi_thr, window)
    df_ts_sweep.to_csv(datadir / "sweep_timeseries.csv", index=False)
    df_sum_sweep.to_csv(datadir / "sweep_summary.csv", index=False)
    point_df, pair_df, sweep_summary = compute_sweep_stats(df_sum_sweep)
    point_df.to_csv(datadir / "sweep_point_estimates.csv", index=False)
    pair_df.to_csv(datadir / "sweep_pairwise_tests.csv", index=False)
    with open(datadir / "sweep_stat_tests.json", "w") as f:
        json.dump({
            "summary": sweep_summary,
            "mi_threshold": float(mi_thr),
            "focal_inherit_for_threshold": float(focal_inherit),
            "stab_window": int(window),
            "note": "Pairwise tests compare each overlay to baseline at the same inheritance probability; FDR is corrected within each metric family.",
        }, f, indent=2)

    meta = {
        "script": Path(__file__).name,
        "n_gens": int(params["n_gens"]),
        "n_reps": int(params["n_reps"]),
        "focal_inherit": float(focal_inherit),
        "sweep_inherits": [float(x) for x in inherit_probs],
        "mi_threshold_quantile": 0.90,
        "mi_threshold_value": float(mi_thr),
        "stab_window": int(window),
        "seed_scheme": "seed = 2_000_000 + int(md5(condition)[:8],16) + rep",
    }
    with open(datadir / "lockdata_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[JTB] Lockdata written to", datadir)


def plot_all(datadir: Path = DATADIR, figdir: Path = FIGDIR) -> None:
    overlay_ts = pd.read_csv(datadir / "overlay_timeseries.csv")
    overlay_sum = pd.read_csv(datadir / "overlay_summary.csv")
    sweep_point = pd.read_csv(datadir / "sweep_point_estimates.csv")

    plot_timecourse(overlay_ts, y="fitness", out=figdir / "overlay_fitness_timecourse")
    plot_timecourse(overlay_ts, y="mi", out=figdir / "overlay_mi_timecourse")
    plot_metric_bars(overlay_sum, metric="dwell", ylabel="Semantic dwell", out=figdir / "overlay_dwell_bars")
    plot_pstab_bars(overlay_sum, out=figdir / "overlay_p_stab_bars")
    plot_metric_bars(overlay_sum, metric="fitness_final", ylabel="Final mean fitness", out=figdir / "overlay_fitness_final_bars")
    plot_lag_hist(overlay_sum, out=figdir / "overlay_lag_hist")

    plot_sweep_lines(sweep_point, metric="p_stab", ylabel="Stabilization probability", out=figdir / "sweep_p_stab_vs_inherit", ci_lo="p_stab_ci_lo", ci_hi="p_stab_ci_hi", ylim=(0.0, 1.05))
    plot_sweep_lines(sweep_point, metric="mean_dwell", ylabel="Mean semantic dwell", out=figdir / "sweep_dwell_vs_inherit", ci_lo="mean_dwell_ci_lo", ci_hi="mean_dwell_ci_hi")
    plot_sweep_lines(sweep_point, metric="mean_delta_fitness", ylabel=r"Mean $\Delta$ fitness", out=figdir / "sweep_delta_fitness_vs_inherit", ci_lo="mean_delta_fitness_ci_lo", ci_hi="mean_delta_fitness_ci_hi")
    plot_sweep_lines(sweep_point, metric="mean_delta_mi", ylabel=r"Mean $\Delta$ MI", out=figdir / "sweep_delta_mi_vs_inherit", ci_lo="mean_delta_mi_ci_lo", ci_hi="mean_delta_mi_ci_hi")
    print("[JTB] Figures written to", figdir)


def main() -> None:
    ap = argparse.ArgumentParser(description="JTB semantic stabilization suite")
    ap.add_argument("--run", action="store_true", help="Run simulations and write lockdata")
    ap.add_argument("--plot", action="store_true", help="Plot figures from existing lockdata")
    ap.add_argument("--force", action="store_true", help="Overwrite existing lockdata if present")
    args = ap.parse_args()

    if not args.run and not args.plot:
        raise SystemExit("Nothing to do. Use --run and/or --plot.")

    if args.run:
        if DATADIR.exists() and any(DATADIR.iterdir()) and not args.force:
            raise SystemExit(f"{DATADIR} already contains files. Use --force to overwrite.")
        if args.force and DATADIR.exists():
            for p in DATADIR.glob("*"):
                if p.is_file():
                    p.unlink()
        run_all(DATADIR, FIGDIR)
    if args.plot:
        plot_all(DATADIR, FIGDIR)


if __name__ == "__main__":
    main()
