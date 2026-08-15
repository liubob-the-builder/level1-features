#!/usr/bin/env python3
"""Mask-aware DFT/RL-C9/RL-C6 on the synthetic SIC-Gen gallery, for the missing cell
of the 2x2 (mask-blind vs mask-aware) x (synthetic vs real) comparison.

Reuses the same mask-aware feature paths exercised in eval_myfeatures_realcasia_maskaware.py
(dft_spectrum.code_magnitude_spectrum(..., mask=mask), rl_features.extract_rl_vector(..., mask=mask))
and the same bijective gallery/probe SSD-ranking method as stage2_evaluate.py (ssd_matrix,
ranks_of_true_identity via diagonal, eer, d_prime -- imported unmodified), on
../sic-gen/sicgen_gallery_2000subjects/ -- the same 2000-subject gallery stage2_results.json
was computed from. The mask-blind baseline is loaded from stage2_results.json (not recomputed);
stage2_results.json is not modified.

AC is not recomputed here: it was already mask-aware in the original Stage 2 run (see
ac_features.py), so its synthetic numbers are identical in both conventions by construction --
only DFT, RL-C9, RL-C6 gain a mask-aware path.

Mask convention check (done before running, reported in printed output and metadata): SIC-Gen's
mask is confirmed 1=valid via template.py's Template.add_noise, which builds
self._mask = np.logical_not(arch_mask) -- arch_mask marks the occluded region, so NOT(arch_mask)
is 1 exactly where the bit is valid/unoccluded. This matches the 1=valid convention the
mask-aware DFT/RL code expects, and the convention real CASIA masks already use.
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent  # level1-features/experiments/
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_SICGEN_DIR = _ROOT / "sic-gen" / "sicgen_gallery_2000subjects"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "myfeatures_synthetic_maskaware.json"
_STAGE2_RESULTS_PATH = _RESULTS / "stage2_results.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from dft_spectrum import code_magnitude_spectrum, load_template, MIN_VALID_FRAC  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)  # bins 30-50 inclusive, matching stage2_evaluate.py
FEATURE_NAMES = ("DFT", "RL-C9", "RL-C6")  # AC excluded: already mask-aware, identical by construction
K_VALUES = [10, 50, 100, 300]

MASK_HANDLING = {
    "DFT": f"mask-aware -- code_magnitude_spectrum(template, mask=mask). Rows with valid "
           f"fraction < {MIN_VALID_FRAC} dropped from the average; kept rows mean-filled. "
           "Bins 30-50 kept, same as mask-blind.",
    "RL-C9": "mask-aware -- extract_rl_vector(template, C=9, mask=mask). Run-lengths computed "
             "only within contiguous valid-bit segments.",
    "RL-C6": "mask-aware -- same as RL-C9, C=6.",
}


def load_mask(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def extract_all_features_maskaware(subject_dirs, stem: str):
    dft, rl_c9, rl_c6 = [], [], []
    valid_fracs = []
    for sd in subject_dirs:
        template = load_template(sd, stem)
        mask = load_mask(sd, stem)
        valid_fracs.append(mask.mean())
        dft.append(code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS])
        rl_c9.append(extract_rl_vector(template, C=9, mask=mask).astype(np.float64))
        rl_c6.append(extract_rl_vector(template, C=6, mask=mask).astype(np.float64))
    return {
        "DFT": np.stack(dft),
        "RL-C9": np.stack(rl_c9),
        "RL-C6": np.stack(rl_c6),
    }, valid_fracs


def ranks_of_true_identity(D: np.ndarray) -> np.ndarray:
    rank_matrix = np.argsort(np.argsort(D, axis=1), axis=1)
    n = D.shape[0]
    return rank_matrix[np.arange(n), np.arange(n)] + 1


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"{_OUT_PATH} already exists; write a new filename instead.")
    if not _STAGE2_RESULTS_PATH.exists():
        raise FileNotFoundError(f"expected existing mask-blind synthetic results at {_STAGE2_RESULTS_PATH}")

    stage2 = json.loads(_STAGE2_RESULTS_PATH.read_text())

    subject_dirs = sorted(
        (d for d in _SICGEN_DIR.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects from {_SICGEN_DIR}")
    assert n == stage2["n_subjects"], "gallery size mismatch vs stage2_results.json"

    # ---- Mask sanity check ----
    sample_masks_exist = all((sd / "1_mask.txt").exists() and (sd / "2_mask.txt").exists() for sd in subject_dirs[:20])
    print(f"Mask files exist (spot check, 20 subjects): {sample_masks_exist}")

    print("\nExtracting gallery (IC1) mask-aware features ...")
    gallery_feats, gallery_valid_fracs = extract_all_features_maskaware(subject_dirs, "1")
    print("Extracting probe (IC2) mask-aware features ...")
    probe_feats, probe_valid_fracs = extract_all_features_maskaware(subject_dirs, "2")

    all_valid_fracs = np.array(gallery_valid_fracs + probe_valid_fracs)
    mean_valid_frac = float(all_valid_fracs.mean())
    print(f"\nSynthetic mask valid-fraction (1=valid, confirmed via template.py's "
          f"np.logical_not(arch_mask) construction): mean={mean_valid_frac:.4f}, "
          f"min={all_valid_fracs.min():.4f}, max={all_valid_fracs.max():.4f}, "
          f"n_masks={len(all_valid_fracs)}")
    print("For comparison, real CASIA mean valid-fraction was ~0.677 (see "
          "myfeatures_realcasia_maskaware.json).")

    results = {}
    for name in FEATURE_NAMES:
        gallery = gallery_feats[name]
        probe = probe_feats[name]
        D = ssd_matrix(gallery, probe)
        ranks = ranks_of_true_identity(D)

        genuine = np.diag(D)
        impostor = D[~np.eye(n, dtype=bool)]

        entry = {
            "dim": gallery.shape[1],
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

    # ---- Comparison table: mask-blind (from stage2_results.json) vs mask-aware (this run) ----
    print("\n" + "=" * 100)
    print(f"{'Feature':<8}{'Convention':<12}{'Dim':>6}" +
          "".join(f"{'R@'+str(k):>8}" for k in K_VALUES) +
          f"{'MedRank':>9}{'MeanRank':>10}{'EER':>8}{'d-prime':>9}")
    print("-" * 100)
    comparison = {}
    for name in FEATURE_NAMES:
        blind = stage2["results_table"][name]
        aware = results[name]
        for label, r in (("blind", blind), ("aware", aware)):
            row_vals = "".join(f"{r[f'recall@{k}']:>8.3f}" for k in K_VALUES)
            print(f"{name:<8}{label:<12}{r['dim']:>6}{row_vals}"
                  f"{r['median_rank']:>9.1f}{r['mean_rank']:>10.1f}{r['eer']:>8.4f}{r['d_prime']:>9.3f}")
        comparison[name] = {"mask_blind": blind, "mask_aware": aware}
    print("=" * 100)

    out = {
        "metadata": {
            "gallery": str(_SICGEN_DIR.relative_to(_ROOT)),
            "n_subjects": n,
            "mask_convention": "1=valid, 0=occluded -- confirmed via sic-gen/template.py's Template.add_noise, "
                                "which sets self._mask = np.logical_not(arch_mask) (arch_mask marks the occluded "
                                "region, so the stored mask is 1 exactly where a bit is valid/unoccluded); matches "
                                "the convention real CASIA masks use and that the mask-aware DFT/RL code expects.",
            "synthetic_mask_valid_fraction_mean": mean_valid_frac,
            "synthetic_mask_valid_fraction_min": float(all_valid_fracs.min()),
            "synthetic_mask_valid_fraction_max": float(all_valid_fracs.max()),
            "real_casia_mask_valid_fraction_mean_for_comparison": 0.677,
            "mask_handling": MASK_HANDLING,
            "mask_blind_baseline_source": str(_STAGE2_RESULTS_PATH.relative_to(_ROOT)),
            "ac_note": "AC excluded from this run -- it was already mask-aware in stage2_evaluate.py, "
                       "so its synthetic numbers are identical under both conventions by construction.",
            "dft_min_valid_frac_threshold": MIN_VALID_FRAC,
            "feature_defaults_unchanged": "mask=None in dft_spectrum.py/rl_features.py still reproduces the exact "
                                           "prior mask-blind behavior; stage2_results.json itself was not modified "
                                           "or recomputed by this script.",
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "k_values": K_VALUES,
            "method_note": "Same bijective gallery/probe SSD-ranking method as stage2_evaluate.py "
                            "(ssd_matrix, eer, d_prime imported unmodified; true-identity rank read off "
                            "the SSD matrix diagonal).",
        },
        "results_maskaware": results,
        "results_maskblind": {name: stage2["results_table"][name] for name in FEATURE_NAMES},
        "comparison": comparison,
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
