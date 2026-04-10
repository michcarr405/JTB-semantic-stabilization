"""Replication-like overlays for Paper 3.

Design constraints:
- Model B (Paper 1) is conceptually frozen.
- Noisy inheritance (random replacement / mutation at division) remains in place.
- Overlays act as *additional mechanisms* that can preserve structure under reset,
  without directly changing the nominal inherit_prob.

We implement three overlays on the Paper-1 reproduction step:

TAC (template-assisted copying):
  When a would-be random sequence is created (non-inherited branch), with probability
  `strength` we instead create a noisy copy of a parental sequence (template copy).
  This increases redundancy and combats stochastic loss at division.

PB (partition bias):
  When sequences are inherited, with probability `corr` we reuse the same parental
  sequence index for all inherited sequences in a daughter cell. This preserves
  co-occurrence structure across division.

AR (autocatalytic reinforcement):
  When a would-be random sequence is created (non-inherited branch), instead of
  sampling nucleotides uniformly, we sample from a convex combination of uniform and
  the parental nucleotide frequencies (strength controls mixing). This biases the
  'random' branch toward reconstituting parental compositional structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OverlayConfig:
    kind: str  # "none" | "TAC" | "PB" | "AR"
    strength: float = 0.0
    error: float = 0.0
    corr: float = 0.0


def _mutate_seq(seq: np.ndarray, mu: float, rng: np.random.Generator) -> np.ndarray:
    # Paper 1 mutation model: per-position random substitution to a different base
    # Implemented here to keep overlays consistent with Paper 1.
    if mu <= 0:
        return seq
    for pos in range(seq.shape[0]):
        if rng.random() < mu:
            old = int(seq[pos])
            seq[pos] = (old + int(rng.integers(1, 4))) % 4
    return seq


def apply_overlay_to_reproduction(
    *,
    pop: np.ndarray,
    fitnesses: np.ndarray,
    mu: float,
    inherit_prob: float,
    overlay: OverlayConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Paper-1 reproduction_with_partitioning with overlay hooks.

    This is a faithful re-implementation of Paper-1 reproduce_with_partitioning,
    with the ONLY differences being:
      - TAC/AR: modify *only* the non-inherited branch's sequence generation.
      - PB: modify *only* which parent sequence index is chosen when inheriting.

    The nominal inherit_prob is unchanged.
    """

    fitnesses = np.maximum(fitnesses, 0)
    probs = fitnesses / fitnesses.sum() if fitnesses.sum() > 0 else np.full_like(fitnesses, 1 / len(fitnesses), dtype=float)

    n_cells, n_seqs, seq_len = pop.shape
    new_pop = np.empty_like(pop)

    kind = overlay.kind
    s = float(overlay.strength)
    err = float(overlay.error)
    corr = float(overlay.corr)

    # Precompute nucleotide frequencies per potential parent cell for AR.
    # Only needed if AR is active and non-inherited branch triggers.
    # We'll compute per chosen parent for simplicity.

    for i in range(n_cells):
        parent = int(rng.choice(n_cells, p=probs))
        parent_seqs = pop[parent]

        # PB: choose a shared parent sequence index (within this chosen parent cell)
        shared_parent_seq_idx: Optional[int] = None
        if kind == "PB" and corr > 0 and rng.random() < corr:
            shared_parent_seq_idx = int(rng.integers(0, n_seqs))

        # AR: compute parent nucleotide distribution over all bases
        parent_base_p = None
        if kind == "AR" and s > 0:
            counts = np.bincount(parent_seqs.reshape(-1), minlength=4).astype(float)
            total = counts.sum()
            parent_base_p = counts / total if total > 0 else np.full(4, 0.25)

        for s_idx in range(n_seqs):
            if rng.random() < inherit_prob:
                # inherited branch: copy a parental sequence
                if shared_parent_seq_idx is not None:
                    src_idx = shared_parent_seq_idx
                else:
                    src_idx = int(rng.integers(0, n_seqs))
                seq = np.array(parent_seqs[src_idx], copy=True)
            else:
                # non-inherited branch
                if kind == "TAC" and s > 0 and rng.random() < s:
                    # Template-assisted copy of a parental sequence with additional copying error.
                    src_idx = int(rng.integers(0, n_seqs))
                    seq = np.array(parent_seqs[src_idx], copy=True)

                    # "copying error" is an extra per-position substitution probability.
                    if err > 0:
                        for pos in range(seq_len):
                            if rng.random() < err:
                                old = int(seq[pos])
                                seq[pos] = (old + int(rng.integers(1, 4))) % 4
                elif kind == "AR" and s > 0 and parent_base_p is not None:
                    # Autocatalytic reinforcement: sample bases from a mixture of uniform and parent composition.
                    # strength s mixes parent composition into the random branch.
                    # error parameter can weaken reinforcement (optional).
                    mix = max(0.0, min(1.0, s))
                    if err > 0:
                        mix *= max(0.0, 1.0 - err)
                    p = (1.0 - mix) * np.array([0.25, 0.25, 0.25, 0.25]) + mix * parent_base_p
                    # numerical safety
                    p = p / p.sum()
                    seq = rng.choice(4, size=seq_len, p=p).astype(np.int8)
                else:
                    # original Paper-1 random sequence
                    seq = rng.integers(0, 4, size=seq_len, dtype=np.int8)

            # Paper-1 mutation
            seq = _mutate_seq(seq, mu, rng)
            new_pop[i, s_idx] = seq

    return new_pop
