"""
Plotting utilities for Model B sweeps.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ============================================================
# BAR PLOTS
# ============================================================

def plot_bar(x, means, sems, ylabel, title, xlim=None, ylim=None):
    plt.figure(figsize=(9, 6))
    plt.bar(np.arange(len(x)), means, yerr=sems, alpha=0.7, capsize=4)
    plt.xticks(np.arange(len(x)), x)

    plt.xlabel("Inheritance Fidelity", fontsize=25)
    plt.ylabel(ylabel, fontsize=25)
    plt.title(title, fontsize=25)

    plt.tick_params(axis='both', labelsize=15)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


# ============================================================
# COLOR PALETTE
# ============================================================

def _tol_colors(n):
    base = [
        "#332288", "#88CCEE", "#44AA99", "#117733",
        "#999933", "#DDCC77", "#CC6677", "#882255", "#AA4499"
    ]
    return [base[i % len(base)] for i in range(n)]


# ============================================================
# MUTUAL INFORMATION TRAJECTORIES
# ============================================================

def plot_MI_trajectories_all(
    MI_results,
    inherit_probs,
    title_suffix="selective interaction",
    ylim=None
):
    plt.figure(figsize=(9, 6))
    colors = _tol_colors(len(inherit_probs))
    gens = None

    for color, inh in zip(colors, inherit_probs):
        histories = MI_results[inh]
        mean_traj = histories.mean(axis=0)
        sem_traj = histories.std(axis=0) / np.sqrt(histories.shape[0])

        if gens is None:
            gens = np.arange(mean_traj.shape[0])

        plt.plot(gens, mean_traj, color=color, label=f"inh={inh:.1f}")
        plt.fill_between(
            gens,
            mean_traj - sem_traj,
            mean_traj + sem_traj,
            color=color,
            alpha=0.25
        )

    plt.xlabel("Generation", fontsize=25)
    plt.ylabel("Mutual Information (bits)", fontsize=25)
    plt.title(f"\n{title_suffix}", fontsize=25)

    plt.tick_params(axis='both', labelsize=15)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend(
        title="Inheritance Probability",
        fontsize=15,
        title_fontsize=18,
        frameon=False,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# FITNESS TRAJECTORIES
# ============================================================

def plot_fitness_trajectories_all(
    fit_results,
    inherit_probs,
    title_suffix="non-selective interaction",
    xlim=None,
    ylim=None
):
    plt.figure(figsize=(9, 6))
    colors = _tol_colors(len(inherit_probs))
    gens = None

    for color, inh in zip(colors, inherit_probs):
        histories = fit_results[inh]
        mean_traj = histories.mean(axis=0)
        sem_traj = histories.std(axis=0) / np.sqrt(histories.shape[0])

        if gens is None:
            gens = np.arange(mean_traj.shape[0])

        plt.plot(gens, mean_traj, color=color, label=f"inh={inh:.1f}")
        plt.fill_between(
            gens,
            mean_traj - sem_traj,
            mean_traj + sem_traj,
            color=color,
            alpha=0.25
        )

    plt.xlabel("Generation", fontsize=25)
    plt.ylabel("Mean Fitness", fontsize=25)
    plt.title(f"\n{title_suffix}", fontsize=25)

    plt.tick_params(axis='both', labelsize=15)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(
        title="Inheritance Probability",
        fontsize=15,
        title_fontsize=18,
        frameon=False,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()