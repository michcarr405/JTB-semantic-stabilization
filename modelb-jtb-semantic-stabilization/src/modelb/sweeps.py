"""
Inheritance sweeps and statistical summaries.
"""

import numpy as np
from tqdm import tqdm
from .simulation import run_single_sim

def mean_sem(arr):
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = std / np.sqrt(len(arr))
    return mean, sem

def permutation_test_delta(deltas, n_perm=5000, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    obs = np.mean(deltas)
    n = len(deltas)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1,1], size=n)
        if abs(np.mean(deltas*signs)) >= abs(obs):
            count += 1
    return obs, (count+1)/(n_perm+1)

def run_inheritance_sweep(
    inherit_probs,
    motif_affinity_matrix,
    params,
    label="selective interaction",
    seed_offset=0
):
    fit_results = {}
    MI_results = {}
    n_reps = params["n_reps"]
    mu = params["mutation_rate"]

    for inh in inherit_probs:
        print(f"Running inherit_prob={inh:.2f} ({label}) ...")
        fit_histories = []
        MI_histories = []

        for r in tqdm(range(n_reps)):
            seed = seed_offset + 1000 + r
            fit_hist, MI_hist = run_single_sim(
                mu,
                inh,
                seed,
                motif_affinity_matrix,
                params
            )
            fit_histories.append(fit_hist)
            MI_histories.append(MI_hist)

        fit_results[inh] = np.vstack(fit_histories)
        MI_results[inh] = np.vstack(MI_histories)

    return fit_results, MI_results

def summarize_inheritance_sweep(fit_results, MI_results, inherit_probs, label="selective interaction"):
    print(f"\n=== INHERITANCE SWEEP SUMMARY ({label}) ===")
    print("inh_prob\tΔFitness_mean\t[95%CI]\tp_fit\tΔMI_mean\t[95%CI]\tp_MI")

    final_dfits = []
    final_dmis = []

    for inh in inherit_probs:
        fits = fit_results[inh]
        MIs = MI_results[inh]

        dfit = fits[:,-1] - fits[:,0]
        dmi = MIs[:,-1] - MIs[:,0]

        mean_dfit, sem_dfit = mean_sem(dfit)
        ci_dfit = (mean_dfit-1.96*sem_dfit, mean_dfit+1.96*sem_dfit)
        _, pfit = permutation_test_delta(dfit)

        mean_dmi, sem_dmi = mean_sem(dmi)
        ci_dmi = (mean_dmi-1.96*sem_dmi, mean_dmi+1.96*sem_dmi)
        _, pmi = permutation_test_delta(dmi)

        final_dfits.append(mean_dfit)
        final_dmis.append(mean_dmi)

        print(f"{inh:.2f}\t{mean_dfit:.4f}\t[{ci_dfit[0]:.4f},{ci_dfit[1]:.4f}]\t"
              f"{pfit:.4f}\t{mean_dmi:.4f}\t[{ci_dmi[0]:.4f},{ci_dmi[1]:.4f}]\t{pmi:.4f}")

    return np.array(final_dfits), np.array(final_dmis)
