#!/usr/bin/env python3
"""Stage 2 — full head-to-head evaluation of DFT / AC / RL Level-1 features.

Gallery/probe split (mirrors the paper): IC1 = gallery, IC2 = genuine probe,
per subject, on sicgen_gallery_2000subjects/.

Features (dimensions fixed by prior agreement with the user):
  DFT: angular DFT-magnitude, bins 30-50 (21-dim). Bin set chosen in the
       Stage 1 signal check; the 78-82 harmonic added negligible d' and was
       dropped.
  AC:  ac_features.py 'concat' mode -- the 5 paper lag-energy bins per row,
       concatenated across all 32 rows (160-dim). The paper states 90-dim
       for this feature; that number doesn't follow from "5 bins x 32 rows"
       under any described aggregation (flagged in ac_features.py's own
       docstring). Per explicit user instruction, we use the existing
       'concat' implementation as-is and report its true dimension (160)
       rather than forcing a dimension we can't derive from the paper text.
  RL-C9: rl_features.py, C=9 -> 2*C=18 bins/row x 32 rows = 576-dim (matches
       the paper's stated dimension and the "18 bins/row" the user originally
       specified; "C=6" does not reproduce 18 bins/row under this code's C
       convention, so C=9 was used first to hit the explicit 576-dim/18
       bins-per-row target).
  RL-C6: rl_features.py, C=6 -> 2*C=12 bins/row x 32 rows = 384-dim. Added
       alongside RL-C9 per explicit user request, following the paper's
       described construction (C=6) literally rather than its stated 576
       dimension, so the two RL variants can be compared side by side.

For each feature: rank the true identity by SSD for every probe, and report
Recall@10/50/100/300, median rank, mean rank, Level-1 EER, d', and dimension.

Also reports the error-correlation analysis: Spearman rank correlation of
per-probe true-identity ranks between each pair of features, and Jaccard
overlap of each feature's failure set (true rank > 50), for every pair among
the four features.

No fusion is performed here.
"""

import itertools
import json
import sys
from pathlib import Path
# Anchor to repo layout regardless of working directory:
# this file is in level1-features/experiments/ ; galleries are in ../../sic-gen/,
# feature modules are in ../features/
_SICGEN = Path(__file__).resolve().parent.parent.parent / "sic-gen"
_FEATURES = Path(__file__).resolve().parent.parent / "features"
sys.path.insert(0, str(_FEATURES))

import numpy as np
from scipy.stats import spearmanr

from ac_features import extract_ac_vector
from dft_spectrum import code_magnitude_spectrum, load_template
from rl_features import extract_rl_vector

GALLERY_DIR = _SICGEN / "sicgen_gallery_2000subjects"
DFT_BINS = np.arange(30, 51)  # bins 30-50 inclusive, per Stage 1 signal check
FAILURE_RANK_THRESHOLD = 50
FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")


def load_mask(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def extract_all_features(subject_dirs, stem: str):
    """Returns dict feature_name -> (n_subjects, dim) array, for IC `stem`."""
    dft, ac, rl_c9, rl_c6 = [], [], [], []
    for sd in subject_dirs:
        template = load_template(sd, stem)
        mask = load_mask(sd, stem)
        dft.append(code_magnitude_spectrum(template, row_agg="average")[DFT_BINS])
        ac.append(extract_ac_vector(template, mask, mode="concat"))
        rl_c9.append(extract_rl_vector(template, C=9).astype(np.float64))
        rl_c6.append(extract_rl_vector(template, C=6).astype(np.float64))
    return {
        "DFT": np.stack(dft),
        "AC": np.stack(ac),
        "RL-C9": np.stack(rl_c9),
        "RL-C6": np.stack(rl_c6),
    }


def ssd_matrix(gallery: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """D[i, j] = SSD(probe_i, gallery_j), via the ||a||^2+||b||^2-2ab identity
    to avoid an O(n^2 * dim) memory blowup for the 576-dim RL feature."""
    g_sq = np.sum(gallery ** 2, axis=1)
    p_sq = np.sum(probe ** 2, axis=1)
    cross = probe @ gallery.T
    D = p_sq[:, None] + g_sq[None, :] - 2 * cross
    return np.maximum(D, 0.0)


def ranks_of_true_identity(D: np.ndarray) -> np.ndarray:
    """For each probe i (row), the 1-indexed rank of gallery entry i (ascending SSD)."""
    rank_matrix = np.argsort(np.argsort(D, axis=1), axis=1)
    n = D.shape[0]
    return rank_matrix[np.arange(n), np.arange(n)] + 1


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    gen_sorted = np.sort(genuine)
    imp_sorted = np.sort(impostor)
    thresholds = np.unique(np.concatenate([gen_sorted, imp_sorted]))
    far = np.searchsorted(imp_sorted, thresholds, side="right") / len(imp_sorted)
    frr = 1.0 - np.searchsorted(gen_sorted, thresholds, side="right") / len(gen_sorted)
    idx = np.argmin(np.abs(far - frr))
    return float((far[idx] + frr[idx]) / 2)


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return float(abs(impostor.mean() - genuine.mean()) / np.sqrt(0.5 * (genuine.var() + impostor.var())))


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def main():
    subject_dirs = sorted(
        (d for d in GALLERY_DIR.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects from {GALLERY_DIR}")

    print("Extracting gallery (IC1) features ...")
    gallery_feats = extract_all_features(subject_dirs, "1")
    print("Extracting probe (IC2) features ...")
    probe_feats = extract_all_features(subject_dirs, "2")

    results = {}
    ranks_by_feature = {}
    D_by_feature = {}

    for name in FEATURE_NAMES:
        gallery = gallery_feats[name]
        probe = probe_feats[name]
        D = ssd_matrix(gallery, probe)
        D_by_feature[name] = D

        ranks = ranks_of_true_identity(D)
        ranks_by_feature[name] = ranks

        genuine = np.diag(D)
        impostor = D[~np.eye(n, dtype=bool)]

        results[name] = {
            "dim": gallery.shape[1],
            "recall@10": recall_at_k(ranks, 10),
            "recall@50": recall_at_k(ranks, 50),
            "recall@100": recall_at_k(ranks, 100),
            "recall@300": recall_at_k(ranks, 300),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "eer": eer(genuine, impostor),
            "d_prime": d_prime(genuine, impostor),
        }

    # ---- Results table ----
    print("\n" + "=" * 100)
    print(f"{'Feature':<8}{'Dim':>6}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'R@300':>8}"
          f"{'MedRank':>10}{'MeanRank':>10}{'EER':>8}{'d-prime':>9}")
    print("-" * 100)
    for name in FEATURE_NAMES:
        r = results[name]
        print(f"{name:<8}{r['dim']:>6}{r['recall@10']:>8.3f}{r['recall@50']:>8.3f}"
              f"{r['recall@100']:>8.3f}{r['recall@300']:>8.3f}{r['median_rank']:>10.1f}"
              f"{r['mean_rank']:>10.1f}{r['eer']:>8.4f}{r['d_prime']:>9.3f}")
    print("=" * 100)

    # ---- Error-correlation analysis ----
    print("\nSpearman rank correlation of per-probe true-identity ranks:")
    pairs = list(itertools.combinations(FEATURE_NAMES, 2))
    spearman_results = {}
    for a, b in pairs:
        rho, pval = spearmanr(ranks_by_feature[a], ranks_by_feature[b])
        spearman_results[f"{a}-{b}"] = {"rho": float(rho), "pval": float(pval)}
        print(f"  {a} vs {b}: rho={rho:.4f}  (p={pval:.2e})")

    print(f"\nFailure-set Jaccard overlap (true rank > {FAILURE_RANK_THRESHOLD}):")
    failure_sets = {
        name: set(np.where(ranks_by_feature[name] > FAILURE_RANK_THRESHOLD)[0].tolist())
        for name in FEATURE_NAMES
    }
    for name, fs in failure_sets.items():
        print(f"  {name}: {len(fs)} failing probes ({len(fs)/n:.1%})")
    jaccard_results = {}
    for a, b in pairs:
        sa, sb = failure_sets[a], failure_sets[b]
        union = sa | sb
        jac = len(sa & sb) / len(union) if union else float("nan")
        jaccard_results[f"{a}-{b}"] = jac
        print(f"  {a} vs {b}: |A|={len(sa)}  |B|={len(sb)}  |A^B|={len(sa & sb)}  |AUB|={len(union)}  Jaccard={jac:.3f}")

    # ---- Save everything for reference ----
    out = {
        "n_subjects": n,
        "results_table": results,
        "spearman": spearman_results,
        "jaccard_failure_overlap": jaccard_results,
        "failure_set_sizes": {k: len(v) for k, v in failure_sets.items()},
    }
    with open("stage2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.savez(
        "stage2_ranks.npz",
        **{f"ranks_{name.replace('-', '_')}": ranks_by_feature[name] for name in FEATURE_NAMES},
    )
    print("\nSaved stage2_results.json and stage2_ranks.npz")


if __name__ == "__main__":
    main()
