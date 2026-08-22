#!/usr/bin/env python3
"""Head-to-head comparison of all Level-1 features -- hand-designed (DFT/AC/RL-C9/RL-C6)
and Christina's enhanced-RL feature -- against the learned CircularConvEncoder (v3), all
evaluated on the EXACT SAME held-out split used for the NN's final test evaluation
(nn_train.py --final-eval), for a fair apples-to-apples comparison.

Matched protocol (must mirror nn_train.py's final-eval / nn_eval_quantized.py exactly):
  - Test probes: only the 299 held-out test identities from
    level1-features/results/nn_identity_split.json (identities.test), every available
    probe stem (2-10) per identity -- NOT all 1998 identities as probes.
  - Gallery: all 1998 usable identities (train+val+test, stem '1'), same pool
    nn_train.py's final-eval ranks test probes against.
  - Exclusions: 524_L/668_L are already absent from nn_identity_split.json (built by
    nn_split.py, which excludes them for a missing gallery image), so no extra
    filtering is needed here.

Hand-designed features: same extraction / mask-handling / SSD ranking as
eval_myfeatures_realcasia_full2000.py (mask-aware DFT bins 30-50, AC concat, RL-C9,
RL-C6), just restricted to this probe/gallery pool instead of all 16,457 probes.

Christina's feature: same extraction / L1-distance ranking as
run_christina_rl_casia_full2000.py (compute_template_features + compute_l1_distance_plaintext,
FilterFHEConfig(), (32,512) codes fed unreshaped from casia-codes-christina/), run under
both mask conventions -- INVERTED (fixed convention, her feature at its best) is the
primary number; RAW is reported alongside for reference only.

NN numbers are not recomputed here -- they're read directly from
level1-features/results/nn_level1_v3_train_history.json's final_test_metrics block,
which was produced by this exact probe/gallery protocol.
"""

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CHRISTINA = _ROOT / "christina-fhe-fis"
_CASIA_2D_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_CASIA_CHRISTINA_DIR = _ROOT / "casia-extraction" / "casia-codes-christina"
_RESULTS = _LEVEL1 / "results"
_SPLIT_PATH = _RESULTS / "nn_identity_split.json"
_NN_HISTORY_PATH = _RESULTS / "nn_level1_v3_train_history.json"
_OUT_PATH = _RESULTS / "all_features_nn_testsplit_comparison.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_CHRISTINA))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template, MIN_VALID_FRAC  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified
from nn_split import load_split  # noqa: E402

from filter_fhe_iris_complete import (  # noqa: E402
    FilterFHEConfig,
    compute_l1_distance_plaintext,
    compute_template_features,
    extract_feature_vector_all_scales,
)

DFT_BINS = np.arange(30, 51)
HAND_FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
K_VALUES = [10, 50, 100, 300]
PROBE_STEMS = [str(i) for i in range(2, 11)]
CHRISTINA_CONVENTIONS = ("inverted", "raw")  # inverted first/primary, raw for reference
EXPECTED_N_TEST_IDENTITIES = 299
EXPECTED_N_GALLERY = 1998

MASK_HANDLING = {
    "DFT": f"mask-aware -- code_magnitude_spectrum(template, mask=mask). Per row: if valid "
           f"fraction < {MIN_VALID_FRAC} the row is dropped from the across-row average; kept "
           "rows have occluded bits mean-filled with that row's own valid-bit mean before the "
           "FFT. Bins 30-50 kept.",
    "AC": "mask-aware -- extract_ac_vector (already mask-aware: row_autocorrelation only counts "
          "bit-pairs where both positions are valid).",
    "RL-C9": "mask-aware -- extract_rl_vector(template, C=9, mask=mask). Run-lengths computed "
             "only within maximal contiguous runs of valid bits.",
    "RL-C6": "mask-aware -- same extract_rl_vector(..., mask=mask) path as RL-C9, just C=6.",
    "Christina-inverted": "occluded_mask = 1 - native_mask fed to compute_enhanced_run_stats "
                           "(matches non_mask_adjacent_valid's True=occluded expectation) -- "
                           "the fixed/corrected convention, her feature at its best.",
    "Christina-raw": "native mask (1=valid) fed as-is -- her real pipeline's literal, "
                      "unmodified behavior. Reported for reference only.",
}


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def available_probe_stems_2d(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()
    ]


def available_probe_stems_christina(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_code.npy").exists() and (identity_dir / f"{s}_mask.npy").exists()
    ]


def load_mask_2d(subject_dir: Path, stem: str) -> np.ndarray:
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
    return np.argsort(np.argsort(D, axis=1), axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def load_christina_pair(identity_dir: Path, stem: str):
    code = np.load(identity_dir / f"{stem}_code.npy")
    mask = np.load(identity_dir / f"{stem}_mask.npy")
    return code, mask


def mask_for_convention(native_mask: np.ndarray, convention: str) -> np.ndarray:
    if convention == "inverted":
        return 1 - native_mask
    elif convention == "raw":
        return native_mask
    raise ValueError(convention)


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(
            f"{_OUT_PATH} already exists; per project rules, write a new filename instead."
        )

    # ---- Load the NN's split and verify counts match the final-eval protocol ----
    split = load_split(_SPLIT_PATH)
    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = sorted(split["identities"]["test"])
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)

    print(f"Test identities (probes): {len(test_ids)} (expected {EXPECTED_N_TEST_IDENTITIES})")
    print(f"Full gallery identities: {len(full_gallery_ids)} (expected {EXPECTED_N_GALLERY})")
    assert len(test_ids) == EXPECTED_N_TEST_IDENTITIES, \
        f"test identity count mismatch: got {len(test_ids)}"
    assert len(full_gallery_ids) == EXPECTED_N_GALLERY, \
        f"gallery identity count mismatch: got {len(full_gallery_ids)}"
    assert "524_L" not in full_gallery_ids and "668_L" not in full_gallery_ids

    # ---- NN reference numbers (read, not recomputed) ----
    nn_history = json.loads(_NN_HISTORY_PATH.read_text())
    nn_test_metrics = nn_history["final_test_metrics"]
    print(f"\nNN (v3) final_test_metrics loaded from {_NN_HISTORY_PATH.name}: "
          f"n_gallery={nn_test_metrics['n_gallery']}, n_probes={nn_test_metrics['n_probes']}")

    # ================= Hand-designed features (DFT/AC/RL-C9/RL-C6) =================
    print("\n--- Extracting hand-designed features (casia-codes-2d) ---")
    t0 = time.time()

    gallery_ids = []
    gallery_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for identity in full_gallery_ids:
        identity_dir = _CASIA_2D_DIR / identity
        g_template = load_template(identity_dir, "1")
        g_mask = load_mask_2d(identity_dir, "1")
        g_feats = extract_features_maskaware(g_template, g_mask)
        gallery_ids.append(identity)
        for name in HAND_FEATURE_NAMES:
            gallery_vecs[name].append(g_feats[name])

    probe_true_ids = []
    probe_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for identity in test_ids:
        identity_dir = _CASIA_2D_DIR / identity
        for stem in available_probe_stems_2d(identity_dir):
            p_template = load_template(identity_dir, stem)
            p_mask = load_mask_2d(identity_dir, stem)
            p_feats = extract_features_maskaware(p_template, p_mask)
            probe_true_ids.append(identity)
            for name in HAND_FEATURE_NAMES:
                probe_vecs[name].append(p_feats[name])

    n_gallery = len(gallery_ids)
    n_probe = len(probe_true_ids)
    print(f"Extracted {n_gallery} galleries, {n_probe} probes ({time.time()-t0:.1f}s).")

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(n_probe)

    hand_results = {}
    for name in HAND_FEATURE_NAMES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery, probe)

        ranks = dense_ranks(D)[row_idx, true_col] + 1
        genuine = D[row_idx, true_col]
        impostor_mask = np.ones_like(D, dtype=bool)
        impostor_mask[row_idx, true_col] = False
        impostor = D[impostor_mask]

        entry = {
            "dim": int(gallery.shape[1]),
            "n_gallery": n_gallery,
            "n_probes": n_probe,
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "eer": eer(genuine, impostor),
            "d_prime": d_prime(genuine, impostor),
        }
        for k in K_VALUES:
            entry[f"recall_at_{k}"] = recall_at_k(ranks, k)
        hand_results[name] = entry
        print(f"  {name}: dim={entry['dim']}, R@10={entry['recall_at_10']:.4f}, "
              f"R@50={entry['recall_at_50']:.4f}, R@100={entry['recall_at_100']:.4f}, "
              f"R@300={entry['recall_at_300']:.4f}, median_rank={entry['median_rank']:.1f}, "
              f"mean_rank={entry['mean_rank']:.2f}")

    # ================= Christina's feature (both mask conventions) =================
    print("\n--- Extracting Christina's feature (casia-codes-christina) ---")
    config = FilterFHEConfig()
    christina_results = {}

    for convention in CHRISTINA_CONVENTIONS:
        print(f"\nConvention: {convention}")
        t0 = time.time()

        gallery_feats = {}
        for identity in full_gallery_ids:
            identity_dir = _CASIA_CHRISTINA_DIR / identity
            gcode, gmask_native = load_christina_pair(identity_dir, "1")
            gmask = mask_for_convention(gmask_native, convention)
            gallery_feats[identity] = compute_template_features(gcode, gmask)
        c_gallery_ids = list(gallery_feats.keys())

        probe_records = []  # (true_identity, features)
        for identity in test_ids:
            identity_dir = _CASIA_CHRISTINA_DIR / identity
            for stem in available_probe_stems_christina(identity_dir):
                pcode, pmask_native = load_christina_pair(identity_dir, stem)
                pmask = mask_for_convention(pmask_native, convention)
                pfeat = compute_template_features(pcode, pmask)
                probe_records.append((identity, pfeat))

        dim = len(extract_feature_vector_all_scales(next(iter(gallery_feats.values()))))
        print(f"  {len(c_gallery_ids)} galleries, {len(probe_records)} probes, "
              f"dim={dim} (feature extraction: {time.time()-t0:.1f}s)")

        t0 = time.time()
        ranks = np.empty(len(probe_records), dtype=np.int64)
        for i, (true_identity, pfeat) in enumerate(probe_records):
            scores = [(gid, compute_l1_distance_plaintext(pfeat, gallery_feats[gid], config))
                      for gid in c_gallery_ids]
            scores.sort(key=lambda x: x[1])
            ranked_ids = [gid for gid, _ in scores]
            ranks[i] = ranked_ids.index(true_identity) + 1
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    ranked {i+1}/{len(probe_records)} probes ({elapsed:.1f}s elapsed)...")

        entry = {
            "convention": convention,
            "dim": dim,
            "n_gallery": len(c_gallery_ids),
            "n_probes": len(probe_records),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
        }
        for k in K_VALUES:
            entry[f"recall_at_{k}"] = recall_at_k(ranks, k)
        christina_results[convention] = entry
        print(f"  {convention}: R@10={entry['recall_at_10']:.4f}, R@50={entry['recall_at_50']:.4f}, "
              f"R@100={entry['recall_at_100']:.4f}, R@300={entry['recall_at_300']:.4f}, "
              f"median_rank={entry['median_rank']:.1f}, mean_rank={entry['mean_rank']:.2f} "
              f"(ranking: {time.time()-t0:.1f}s)")

    # ================= Combined comparison table =================
    nn_entry = {
        "dim": nn_history["hyperparameters"]["embedding_dim"],
        "n_gallery": nn_test_metrics["n_gallery"],
        "n_probes": nn_test_metrics["n_probes"],
        "median_rank": nn_test_metrics["median_rank"],
        "mean_rank": nn_test_metrics["mean_rank"],
        **{f"recall_at_{k}": nn_test_metrics[f"recall_at_{k}"] for k in K_VALUES},
    }

    combined = dict(hand_results)
    combined["Christina (inverted)"] = christina_results["inverted"]
    combined["Christina (raw)"] = christina_results["raw"]
    combined["NN (v3, learned)"] = nn_entry

    print("\n" + "=" * 118)
    print(f"{'Feature':<24}{'Dim':>6}" + "".join(f"{'R@'+str(k):>9}" for k in K_VALUES) +
          f"{'MedRank':>10}{'MeanRank':>10}")
    print("-" * 118)
    for name, r in combined.items():
        print(f"{name:<24}{r['dim']:>6}" +
              "".join(f"{r[f'recall_at_{k}']:>9.4f}" for k in K_VALUES) +
              f"{r['median_rank']:>10.1f}{r['mean_rank']:>10.2f}")
    print("=" * 118)

    out = {
        "metadata": {
            "purpose": "All Level-1 features (hand-designed + Christina's + the learned NN) "
                       "evaluated on the identical held-out split used for the NN's final test "
                       "evaluation (nn_train.py --final-eval), for a fair apples-to-apples comparison.",
            "split_path": str(_SPLIT_PATH.relative_to(_ROOT)),
            "split_seed": split["metadata"]["seed"],
            "split_val_frac": split["metadata"]["val_frac"],
            "split_test_frac": split["metadata"]["test_frac"],
            "n_test_identities_probes": len(test_ids),
            "n_gallery_identities": n_gallery,
            "excluded_identities": ["524_L", "668_L"],
            "exclusion_reason": "failed Open Iris segmentation for their gallery image; already "
                                 "excluded from nn_identity_split.json by nn_split.py.",
            "hand_designed_data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
            "christina_data_source": str(_CASIA_CHRISTINA_DIR.relative_to(_ROOT)),
            "nn_source": str(_NN_HISTORY_PATH.relative_to(_LEVEL1)),
            "nn_checkpoint": nn_history["checkpoint_path"],
            "nn_hyperparameters": nn_history["hyperparameters"],
            "mask_handling": MASK_HANDLING,
            "dft_min_valid_frac_threshold": MIN_VALID_FRAC,
            "christina_config_weights": {
                "weight_hist_scale0": config.weight_hist_scale0,
                "weight_hist_scale1": config.weight_hist_scale1,
                "weight_row": config.weight_row,
                "weight_spatial": config.weight_spatial,
            },
            "distance_metrics": {
                "DFT/AC/RL-C9/RL-C6/NN": "SSD (sum of squared differences)",
                "Christina": "L1 (compute_l1_distance_plaintext)",
            },
            "primary_christina_convention": "inverted",
            "k_values": K_VALUES,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "results": combined,
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
