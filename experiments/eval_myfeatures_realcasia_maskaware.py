#!/usr/bin/env python3
"""Mask-aware real-CASIA evaluation: DFT/RL-C9/RL-C6 with mask support enabled,
compared against the existing mask-blind real-CASIA run and the synthetic Stage 2 numbers.

This is a companion to eval_myfeatures_realcasia.py, which evaluated all four
Level-1 features exactly as stage2_evaluate.py calls them -- DFT and RL mask-blind,
AC already mask-aware. dft_spectrum.py and rl_features.py were subsequently given
an OPTIONAL mask parameter (default None -> unchanged mask-blind behavior, so all
existing synthetic Stage 1/2/3 results still reproduce exactly). This script exercises
that mask-aware path for all four features on the same real CASIA data
(casia-extraction/casia-codes-2d/) used by eval_myfeatures_realcasia.py, and reports
a three-way comparison: mask-blind-real (loaded from the existing results file,
not recomputed) vs. mask-aware-real (computed here) vs. synthetic (loaded from
stage2_results.json / stage2_ranks.npz, same as before).

Mask-aware extraction choices (see feature-module docstrings for the authoritative
description):
  - DFT (code_magnitude_spectrum(..., mask=mask)): per row, if the valid fraction
    is < 0.5 the row is dropped from the across-row average; kept rows have
    occluded bits mean-filled (with that row's own valid-bit mean) before the FFT.
    Bins 30-50 are kept, same as the mask-blind run.
  - AC (extract_ac_vector): unchanged -- it was already mask-aware (1=valid), so
    its mask-aware numbers here are the same computation as the "blind" run.
  - RL-C9 / RL-C6 (extract_rl_vector(..., mask=mask)): run-lengths are computed
    only within maximal contiguous runs of valid bits; an occluded bit terminates
    the run it interrupts and is not itself counted.

No feature module's DEFAULT behavior was changed -- mask=None still reproduces the
original mask-blind path exactly, so existing synthetic results are unaffected.
"""

import itertools
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_EXPERIMENTS = Path(__file__).resolve().parent  # level1-features/experiments/
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "myfeatures_realcasia_maskaware.json"
_BLIND_RESULTS_PATH = _RESULTS / "myfeatures_realcasia.json"
_STAGE2_RESULTS_PATH = _RESULTS / "stage2_results.json"
_STAGE2_RANKS_PATH = _RESULTS / "stage2_ranks.npz"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template, MIN_VALID_FRAC  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)  # bins 30-50 inclusive, matching stage2_evaluate.py / eval_myfeatures_realcasia.py
FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
K_VALUES = [10, 20, 50, 100, 150, 300]
FAILURE_RANK_THRESHOLD = 50
PROBE_STEMS = [str(i) for i in range(2, 11)]

MASK_HANDLING_MASKAWARE = {
    "DFT": f"mask-aware -- code_magnitude_spectrum(template, mask=mask) called with the mask. "
           f"Per row: if valid fraction < {MIN_VALID_FRAC} the row is dropped from the across-row "
           "average; kept rows have occluded bits mean-filled with that row's own valid-bit mean "
           "before the FFT. Bins 30-50 kept, same as the mask-blind run. Default behavior "
           "(mask=None) is unchanged, so synthetic Stage 1/2/3 results still reproduce exactly.",
    "AC": "mask-aware -- identical computation to the mask-blind run (AC was already mask-aware: "
          "row_autocorrelation only counts bit-pairs where both positions are valid).",
    "RL-C9": "mask-aware -- extract_rl_vector(template, C=9, mask=mask) called with the mask. "
             "Run-lengths are computed only within maximal contiguous runs of valid bits; an "
             "occluded bit terminates the run it interrupts and is not itself counted. Default "
             "behavior (mask=None) is unchanged, so synthetic Stage 1/2/3 results still reproduce exactly.",
    "RL-C6": "mask-aware -- same extract_rl_vector(..., mask=mask) path as RL-C9, just C=6; identical caveat.",
}

MASK_AWARE_NOTE = (
    "DFT and RL-C9/RL-C6 now use their new optional mask parameter on real CASIA data; AC's "
    "numbers here are the same computation as in the mask-blind run (it was already mask-aware). "
    "Feature module defaults (mask=None) are unchanged, so this run does not affect reproducibility "
    "of any existing synthetic result."
)


def load_manifest():
    return json.loads(_CASIA_MANIFEST.read_text())


def available_probe_stems(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()
    ]


def load_mask(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def extract_features_maskaware(template: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
        "RL-C6": extract_rl_vector(template, C=6, mask=mask).astype(np.float64),
    }


def dense_ranks(D: np.ndarray) -> np.ndarray:
    """Per-row dense rank (0-indexed, stable ties), mirroring stage2_evaluate.py's
    `np.argsort(np.argsort(D, axis=1), axis=1)`."""
    return np.argsort(np.argsort(D, axis=1), axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(
            f"{_OUT_PATH} already exists; per project rules, write a new filename instead."
        )
    if not _BLIND_RESULTS_PATH.exists():
        raise FileNotFoundError(f"expected existing mask-blind results at {_BLIND_RESULTS_PATH}")

    blind = json.loads(_BLIND_RESULTS_PATH.read_text())

    manifest = load_manifest()
    identities = manifest["identities"]
    n_identities = len(identities)
    print(f"Loaded manifest: {n_identities} identities")
    print(f"Subject selection: {manifest['subject_selection_rule']}")

    print("\nMask handling per feature, this run (mask-aware path):")
    for name in FEATURE_NAMES:
        print(f"  {name}: {MASK_HANDLING_MASKAWARE[name]}")

    # ---- Extraction (mask-aware) ----
    gallery_ids = []
    gallery_vecs = {name: [] for name in FEATURE_NAMES}
    probe_true_ids = []
    probe_vecs = {name: [] for name in FEATURE_NAMES}

    for ident in identities:
        identity = ident["identity"]
        if not ident["gallery_ok"]:
            continue
        identity_dir = _CASIA_2D_DIR / identity

        g_template = load_template(identity_dir, "1")
        g_mask = load_mask(identity_dir, "1")
        g_feats = extract_features_maskaware(g_template, g_mask)
        gallery_ids.append(identity)
        for name in FEATURE_NAMES:
            gallery_vecs[name].append(g_feats[name])

        for stem in available_probe_stems(identity_dir):
            p_template = load_template(identity_dir, stem)
            p_mask = load_mask(identity_dir, stem)
            p_feats = extract_features_maskaware(p_template, p_mask)
            probe_true_ids.append(identity)
            for name in FEATURE_NAMES:
                probe_vecs[name].append(p_feats[name])

    n_gallery = len(gallery_ids)
    n_probe = len(probe_true_ids)
    print(f"\nExtracted {n_gallery} galleries, {n_probe} probes.")

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(n_probe)

    results = {}
    ranks_by_feature = {}

    for name in FEATURE_NAMES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery, probe)  # (n_probe, n_gallery), D[i,j] = SSD(probe_i, gallery_j)

        rank_matrix = dense_ranks(D)
        ranks = rank_matrix[row_idx, true_col] + 1  # 1-indexed
        ranks_by_feature[name] = ranks

        genuine = D[row_idx, true_col]
        impostor_mask = np.ones_like(D, dtype=bool)
        impostor_mask[row_idx, true_col] = False
        impostor = D[impostor_mask]

        entry = {
            "dim": gallery.shape[1],
            "n_gallery": n_gallery,
            "n_probe": n_probe,
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "eer": eer(genuine, impostor),
            "d_prime": d_prime(genuine, impostor),
        }
        for k in K_VALUES:
            entry[f"recall@{k}"] = recall_at_k(ranks, k)
        results[name] = entry

        print(f"  {name}: dim={entry['dim']}, R@10={entry['recall@10']:.3f}, "
              f"R@50={entry['recall@50']:.3f}, median_rank={entry['median_rank']:.1f}")

    # ---- Error-correlation analysis (real data, mask-aware) ----
    pairs = list(itertools.combinations(FEATURE_NAMES, 2))
    spearman_results = {}
    for a, b in pairs:
        rho, pval = spearmanr(ranks_by_feature[a], ranks_by_feature[b])
        spearman_results[f"{a}-{b}"] = {"rho": float(rho), "pval": float(pval)}

    failure_sets = {
        name: set(np.where(ranks_by_feature[name] > FAILURE_RANK_THRESHOLD)[0].tolist())
        for name in FEATURE_NAMES
    }
    jaccard_results = {}
    for a, b in pairs:
        sa, sb = failure_sets[a], failure_sets[b]
        union = sa | sb
        jac = len(sa & sb) / len(union) if union else float("nan")
        jaccard_results[f"{a}-{b}"] = jac

    print("\nSpearman rank correlation (real CASIA, mask-aware):")
    for pair, v in spearman_results.items():
        print(f"  {pair}: rho={v['rho']:.4f} (p={v['pval']:.2e})")
    print(f"\nFailure-set Jaccard overlap (rank > {FAILURE_RANK_THRESHOLD}), real CASIA, mask-aware:")
    for pair, jac in jaccard_results.items():
        print(f"  {pair}: {jac:.3f}")

    print("\nDFT-AC and AC-RL error correlation, mask-blind (loaded) vs mask-aware (this run):")
    for pair in ("DFT-AC", "AC-RL-C9", "AC-RL-C6"):
        blind_rho = blind["spearman_real"][pair]["rho"]
        aware_rho = spearman_results[pair]["rho"]
        blind_jac = blind["jaccard_failure_overlap_real"][pair]
        aware_jac = jaccard_results[pair]
        print(f"  {pair}: spearman blind={blind_rho:.4f} -> aware={aware_rho:.4f}; "
              f"jaccard blind={blind_jac:.3f} -> aware={aware_jac:.3f}")

    # ---- Synthetic Stage 2 comparison (2000-subject SIC-Gen gallery; features run mask-blind there) ----
    stage2 = json.loads(_STAGE2_RESULTS_PATH.read_text())
    stage2_ranks_npz = np.load(_STAGE2_RANKS_PATH)
    synthetic_table = {}
    for name in FEATURE_NAMES:
        npz_key = f"ranks_{name.replace('-', '_')}"
        synth_ranks = stage2_ranks_npz[npz_key]
        base = dict(stage2["results_table"][name])  # dim, recall@10/50/100/300, median/mean rank, eer, d_prime
        for k in K_VALUES:
            base[f"recall@{k}"] = recall_at_k(synth_ranks, k)  # recompute so 20/150 are present too
        synthetic_table[name] = base

    # ---- Combined three-column table: real-blind vs real-aware vs synthetic ----
    print("\n" + "=" * 140)
    print(f"{'Feature':<8}{'Dataset':<14}{'Dim':>6}" +
          "".join(f"{'R@'+str(k):>8}" for k in K_VALUES) +
          f"{'MedRank':>9}{'MeanRank':>10}{'EER':>8}{'d-prime':>9}")
    print("-" * 140)
    for name in FEATURE_NAMES:
        rows = (
            ("real-blind", blind["results_real"][name]),
            ("real-aware", results[name]),
            ("synth", synthetic_table[name]),
        )
        for dataset, r in rows:
            print(f"{name:<8}{dataset:<14}{r['dim']:>6}" +
                  "".join(f"{r[f'recall@{k}']:>8.3f}" for k in K_VALUES) +
                  f"{r['median_rank']:>9.1f}{r['mean_rank']:>10.1f}{r['eer']:>8.4f}{r['d_prime']:>9.3f}")
    print("=" * 140)

    out = {
        "metadata": {
            "data_source_real": "casia-extraction/casia-codes-2d/ (real CASIA-Iris-Thousand via Open Iris)",
            "data_source_synthetic": "level1-features/results/stage2_results.json + stage2_ranks.npz "
                                      "(SIC-Gen sicgen_gallery_2000subjects, run mask-blind -- NOTE: 2000 "
                                      "subjects, not 300; recall@K is not directly comparable across "
                                      "different gallery sizes N)",
            "mask_blind_comparison_source": str(_BLIND_RESULTS_PATH.relative_to(_ROOT)),
            "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
            "real_identity_count": n_identities,
            "real_gallery_size": n_gallery,
            "real_probe_count": n_probe,
            "subject_selection_rule": manifest["subject_selection_rule"],
            "synthetic_gallery_size": stage2["n_subjects"],
            "mask_handling": MASK_HANDLING_MASKAWARE,
            "mask_handling_note": MASK_AWARE_NOTE,
            "dft_min_valid_frac_threshold": MIN_VALID_FRAC,
            "dft_fill_method": "occluded bits mean-filled with that row's own valid-bit mean before FFT; "
                                "rows below the valid-fraction threshold are dropped from the across-row average",
            "rl_mask_method": "run-lengths computed only within maximal contiguous runs of valid bits; "
                               "an occluded bit terminates the run it interrupts and is not counted",
            "feature_defaults_unchanged": "mask=None still reproduces the exact prior mask-blind behavior "
                                           "in both dft_spectrum.py and rl_features.py; no existing synthetic "
                                           "result is affected by this change.",
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "k_values": K_VALUES,
            "failure_rank_threshold": FAILURE_RANK_THRESHOLD,
            "method_note": (
                "Same feature extraction and SSD-ranking method as eval_myfeatures_realcasia.py, with "
                "DFT/RL-C9/RL-C6 now called through their new optional mask parameter. ssd_matrix, eer, "
                "d_prime imported unmodified from stage2_evaluate.py."
            ),
        },
        "results_real_maskaware": results,
        "results_real_maskblind": blind["results_real"],
        "results_synthetic": synthetic_table,
        "spearman_real_maskaware": spearman_results,
        "jaccard_failure_overlap_real_maskaware": jaccard_results,
        "failure_set_sizes_real_maskaware": {k: len(v) for k, v in failure_sets.items()},
        "spearman_real_maskblind": blind["spearman_real"],
        "jaccard_failure_overlap_real_maskblind": blind["jaccard_failure_overlap_real"],
        "spearman_synthetic": stage2["spearman"],
        "jaccard_failure_overlap_synthetic": stage2["jaccard_failure_overlap"],
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
