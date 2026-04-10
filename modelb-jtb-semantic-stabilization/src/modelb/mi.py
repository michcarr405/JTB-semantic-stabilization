"""
Mutual information utilities for Model B.
"""

import numpy as np

def compute_mutual_information(metabolites, motifs, n_metabolites=4):
    """
    Compute MI(motif; metabolite) in bits.
    """
    motifs = np.asarray(motifs, dtype=np.int64)
    metabolites = np.asarray(metabolites, dtype=np.int64)

    n_motifs = 4 ** 5  # 1024
    joint_counts = np.zeros((n_motifs, n_metabolites), dtype=np.float64)

    for m, k in zip(motifs, metabolites):
        joint_counts[m, k] += 1.0

    total = joint_counts.sum()
    if total == 0:
        return 0.0

    P_joint = joint_counts / total
    P_motif = P_joint.sum(axis=1, keepdims=True)
    P_met = P_joint.sum(axis=0, keepdims=True)

    mask = P_joint > 0
    Pj = P_joint[mask]
    Pm = P_motif.repeat(n_metabolites, axis=1)[mask]
    Pk = P_met.repeat(n_motifs, axis=0)[mask]

    MI_bits = np.sum(Pj * np.log2(Pj / (Pm * Pk)))
    return float(MI_bits)
