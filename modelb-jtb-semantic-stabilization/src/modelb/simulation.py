"""
Single-replicate simulation utilities for Model B.
"""

import numpy as np
from .core import compute_fitnesses_and_observations
from .mi import compute_mutual_information

def init_population(n_cells, n_seqs, seq_len, rng):
    return rng.integers(0, 4, size=(n_cells, n_seqs, seq_len), dtype=np.int8)

def reproduce_with_partitioning(pop, fitnesses, mu, inherit_prob, rng):
    fitnesses = np.maximum(fitnesses, 0)
    probs = fitnesses / fitnesses.sum()

    n_cells, n_seqs, seq_len = pop.shape
    new_pop = np.empty_like(pop)

    for i in range(n_cells):
        parent = rng.choice(n_cells, p=probs)
        parent_seqs = pop[parent]

        for s in range(n_seqs):
            if rng.random() < inherit_prob:
                seq = np.array(parent_seqs[rng.integers(0, n_seqs)], copy=True)
            else:
                seq = rng.integers(0, 4, size=seq_len, dtype=np.int8)

            for pos in range(seq_len):
                if rng.random() < mu:
                    old = seq[pos]
                    seq[pos] = (old + rng.integers(1,4)) % 4

            new_pop[i, s] = seq

    return new_pop

def run_single_sim(
    mu,
    inherit_prob,
    seed,
    motif_affinity_matrix,
    params
):
    rng = np.random.default_rng(seed)

    n_cells = params["n_cells"]
    n_seqs = params["n_seqs"]
    seq_len = params["seq_len"]
    n_metabolites = params["n_metabolites"]
    n_gens = params["n_gens"]

    population = init_population(n_cells, n_seqs, seq_len, rng)
    fitness_hist = []
    MI_hist = []

    for _ in range(n_gens):
        fitnesses, motifs, mets = compute_fitnesses_and_observations(
            population,
            motif_affinity_matrix,
            params["segment_favored_met"],
            params["bias_strength"],
            params["productive_pairs"],
            params["anti_pairs"],
            params["reward_strength"],
            params["penalty_strength"],
            params["temperature"]
        )

        fitness_hist.append(fitnesses.mean())
        MI_hist.append(compute_mutual_information(mets, motifs, n_metabolites))

        population = reproduce_with_partitioning(
            population, fitnesses, mu, inherit_prob, rng
        )

    return np.array(fitness_hist), np.array(MI_hist)
