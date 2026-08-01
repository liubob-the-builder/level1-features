#!/usr/bin/env python3
"""Stage 3 -- Reciprocal Rank Fusion (RRF) of Level-1 features on the
Stage 2 gallery (sicgen_gallery_2000subjects/).

stage2_ranks.npz only stores each probe's *true-identity* rank per feature,
not the full per-candidate rank list every other gallery entry received --
and RRF needs the latter (a candidate's fused score depends on its rank
under every feature, not just whether it was the true match). So this
script re-extracts the same four features (DFT bins 30-50, AC concat-160,
RL C=9/576, RL C=6/384 -- identical construction to stage2_evaluate.py,
nothing changed) and rebuilds the full SSD/rank matrices. It then verifies
its own recomputed true-identity ranks against the saved stage2_ranks.npz
as a correctness check before doing any fusion.

RRF: for each probe, each feature gives every gallery candidate a rank
(1 = closest by SSD). Fused score(candidate) = sum over included features
of 1 / (k + rank), with the standard constant k=60. Candidates are
re-sorted by descending fused score; we report the true identity's
position in that fused ordering.

Fusion combinations (RL means RL-C9 throughout, per instructions):
  DFT+AC, DFT+RL, AC+RL (paper's own combination, as a baseline check),
  DFT+AC+RL.
"""

import itertools
from pathlib import Path

import numpy as np

from stage2_evaluate import (
    GALLERY_DIR,
    extract_all_features,
    ranks_of_true_identity,
    ssd_matrix,
)

RRF_K = 60
FUSION_FEATURES = ("DFT", "AC", "RL-C9")


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def rrf_score_matrix(rank_matrices, k: int = RRF_K) -> np.ndarray:
    total = None
    for R in rank_matrices:
        contrib = 1.0 / (k + R)
        total = contrib if total is None else total + contrib
    return total


def metrics_row(ranks: np.ndarray, n: int) -> dict:
    return {
        "n": n,
        "recall@10": recall_at_k(ranks, 10),
        "recall@50": recall_at_k(ranks, 50),
        "recall@100": recall_at_k(ranks, 100),
        "recall@300": recall_at_k(ranks, 300),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def print_table(rows: dict):
    print("\n" + "=" * 92)
    print(f"{'Name':<16}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'R@300':>8}{'MedRank':>10}{'MeanRank':>10}")
    print("-" * 92)
    for name, m in rows.items():
        print(f"{name:<16}{m['recall@10']:>8.3f}{m['recall@50']:>8.3f}{m['recall@100']:>8.3f}"
              f"{m['recall@300']:>8.3f}{m['median_rank']:>10.1f}{m['mean_rank']:>10.1f}")
    print("=" * 92)


def main():
    subject_dirs = sorted(
        (d for d in GALLERY_DIR.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects from {GALLERY_DIR}")

    print("Re-extracting gallery (IC1) and probe (IC2) features (DFT/AC/RL-C9) ...")
    gallery_feats = extract_all_features(subject_dirs, "1")
    probe_feats = extract_all_features(subject_dirs, "2")

    # ---- Full rank matrices (candidate rank, not just true-identity rank) ----
    rank_matrices = {}
    true_ranks = {}
    for name in FUSION_FEATURES:
        D = ssd_matrix(gallery_feats[name], probe_feats[name])
        R = np.argsort(np.argsort(D, axis=1), axis=1) + 1  # 1-indexed rank per candidate
        rank_matrices[name] = R
        true_ranks[name] = R[np.arange(n), np.arange(n)]

    # ---- Sanity check against stage2_ranks.npz ----
    saved = np.load("stage2_ranks.npz")
    key_map = {"DFT": "ranks_DFT", "AC": "ranks_AC", "RL-C9": "ranks_RL_C9"}
    for name in FUSION_FEATURES:
        if not np.array_equal(true_ranks[name], saved[key_map[name]]):
            raise RuntimeError(f"Recomputed {name} true-identity ranks don't match stage2_ranks.npz")
    print("Recomputed true-identity ranks match stage2_ranks.npz for DFT, AC, RL-C9 -- consistent with Stage 2.")

    # ---- Individual-feature rows (already known from Stage 2, shown here for the side-by-side table) ----
    rows = {}
    for name in FUSION_FEATURES:
        rows[name] = metrics_row(true_ranks[name], n)

    # ---- Fusion combinations ----
    combos = {
        "DFT+AC": ("DFT", "AC"),
        "DFT+RL": ("DFT", "RL-C9"),
        "AC+RL": ("AC", "RL-C9"),
        "DFT+AC+RL": ("DFT", "AC", "RL-C9"),
    }
    fused_ranks = {}
    for combo_name, feats in combos.items():
        S = rrf_score_matrix([rank_matrices[f] for f in feats], k=RRF_K)
        fused_R = ranks_of_true_identity(-S)  # higher RRF score = better -> negate for ascending "distance"
        fused_ranks[combo_name] = fused_R
        rows[combo_name] = metrics_row(fused_R, n)

    print(f"\nRRF constant k={RRF_K}")
    print_table(rows)

    # ---- Fusion gain over the better individual component ----
    print("\nR@50 improvement of each fusion pair over the better of its two components:")
    pair_combos = {k: v for k, v in combos.items() if len(v) == 2}
    for combo_name, feats in pair_combos.items():
        best_component_r50 = max(rows[f]["recall@50"] for f in feats)
        fused_r50 = rows[combo_name]["recall@50"]
        gain = fused_r50 - best_component_r50
        print(f"  {combo_name}: fused R@50={fused_r50:.3f}  best component R@50={best_component_r50:.3f}  "
              f"gain={gain:+.3f}")

    print(f"\nDFT+AC+RL (all three): R@50={rows['DFT+AC+RL']['recall@50']:.3f}  "
          f"best single feature R@50={max(rows[f]['recall@50'] for f in FUSION_FEATURES):.3f}  "
          f"gain={rows['DFT+AC+RL']['recall@50'] - max(rows[f]['recall@50'] for f in FUSION_FEATURES):+.3f}")

    out = {
        "rrf_k": RRF_K,
        "n_subjects": n,
        "results_table": rows,
    }
    import json
    with open("stage3_fusion_results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(
        "stage3_fusion_ranks.npz",
        **{f"fused_{k.replace('+', '_')}": v for k, v in fused_ranks.items()},
    )
    print("\nSaved stage3_fusion_results.json and stage3_fusion_ranks.npz")


if __name__ == "__main__":
    main()
