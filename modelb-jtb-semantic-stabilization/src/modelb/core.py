"""
Numba-accelerated core for fitness computation and metabolite observations.
"""

import numpy as np

# Numba speeds up sweeps, but is optional for this Paper 3 code package.
# Some environments ship with a coverage/numba combination that fails at
# import-time. We fall back to pure Python if numba can't be imported.
try:
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        """Fallback njit decorator.

        Supports both usages:
          @njit
          @njit(...)
        """
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Called as @njit
            return args[0]

        # Called as @njit(...)
        def wrap(fn):
            return fn

        return wrap

@njit
def compute_fitnesses_and_observations(
    population,
    motif_affinity_matrix,
    segment_favored_met,
    bias_strength,
    productive_pairs_matrix,
    anti_pairs_matrix,
    reward_strength,
    penalty_strength,
    temperature
):
    n_cells, n_seqs, seq_len = population.shape
    n_met = motif_affinity_matrix.shape[1]
    chain_len = seq_len - 4

    n_segments = segment_favored_met.shape[0]
    seg_len = chain_len // n_segments

    total_windows = n_cells * n_seqs * chain_len
    motif_obs = np.empty(total_windows, dtype=np.int64)
    met_obs = np.empty(total_windows, dtype=np.int64)

    fitnesses = np.empty(n_cells, dtype=np.float64)
    obs_idx = 0

    for ci in range(n_cells):
        prod_sum = 0.0
        anti_sum = 0.0

        for si in range(n_seqs):
            prev_met = -1
            for pos in range(chain_len):

                motif_idx = 0
                for k in range(5):
                    motif_idx = motif_idx * 4 + int(population[ci, si, pos+k])

                motif_obs[obs_idx] = motif_idx

                logits = motif_affinity_matrix[motif_idx].copy()
                seg_idx = pos // seg_len
                fav = segment_favored_met[seg_idx]
                logits[int(fav)] += bias_strength

                maxval = logits[0]
                for m in range(1, n_met):
                    if logits[m] > maxval:
                        maxval = logits[m]

                sumexp = 0.0
                for m in range(n_met):
                    val = np.exp((logits[m]-maxval)/temperature)
                    logits[m] = val
                    sumexp += val

                r = np.random.random()
                cdf = 0.0
                chosen = 0
                for m in range(n_met):
                    cdf += logits[m]/sumexp
                    if r <= cdf:
                        chosen = m
                        break

                met_obs[obs_idx] = chosen
                obs_idx += 1

                if pos > 0:
                    if productive_pairs_matrix[prev_met, chosen]:
                        prod_sum += 1.0
                    if anti_pairs_matrix[prev_met, chosen]:
                        anti_sum += 1.0

                prev_met = chosen

        mean_prod = prod_sum / n_seqs
        mean_anti = anti_sum / n_seqs
        fit = 1.0 + reward_strength * mean_prod - penalty_strength * mean_anti
        if fit < 0.1:
            fit = 0.1

        fitnesses[ci] = fit

    return fitnesses, motif_obs, met_obs
