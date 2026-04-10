"""
Configuration and default parameters for Model B.
"""

import numpy as np

def default_parameters():
    params = {}

    # population structure
    params["n_cells"] = 80
    params["n_seqs"] = 8
    params["seq_len"] = 60
    params["n_metabolites"] = 4

    # environment / bias
    params["segment_favored_met"] = np.array([2, 0, 3, 1], dtype=np.int64)
    params["bias_strength"] = 2.0

    # fitness parameters
    params["reward_strength"] = 2.0
    params["penalty_strength"] = 0.5
    params["temperature"] = 1.2

    # productive / anti-productive metabolite pairs
    n_met = params["n_metabolites"]
    productive_pairs = np.zeros((n_met, n_met), dtype=np.bool_)
    for a, b in [(0,1), (1,2), (2,3), (3,0)]:
        productive_pairs[a, b] = True
        productive_pairs[b, a] = True
    params["productive_pairs"] = productive_pairs

    anti_pairs = np.zeros((n_met, n_met), dtype=np.bool_)
    for i in range(n_met):
        anti_pairs[i, i] = True
    anti_pairs[1,3] = True
    anti_pairs[3,1] = True
    params["anti_pairs"] = anti_pairs

    # evolution parameters
    params["n_gens"] = 15
    params["n_reps"] = 2
    params["mutation_rate"] = 0.001

    # inheritance probabilities to sweep
    params["inherit_probs"] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    # global RNG for motif affinity initialization
    rng_global = np.random.default_rng(123)
    params["motif_affinity_matrix"] = rng_global.normal(
        0, 1, size=(4**5, params["n_metabolites"])
    )
    params["motif_affinity_matrix_no_aff"] = np.zeros_like(params["motif_affinity_matrix"])

    return params
