#!/usr/bin/env python3
"""De-duplicated re-run of eval_all_features_nn_testsplit.py: excludes the 5 test
probes that are byte-identical to their own gallery image (a CASIA-Iris-Thousand
source-data artifact -- see audit_nn_v3_verification.json check_2 and
audit_nn_v3_dedup_metrics.json), for every feature (hand-designed AND Christina's),
not just the NN. Same extraction/ranking code, same matched split, just 5 fewer
probes going into every feature's evaluation.

Read-only w.r.t. all existing checkpoints/split/results; writes only to a new
filename.
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
_NN_DEDUP_PATH = _RESULTS / "audit_nn_v3_dedup_metrics.json"
_OUT_PATH = _RESULTS / "audit_all_features_dedup_testsplit.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_CHRISTINA))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template, MIN_VALID_FRAC  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402
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
CHRISTINA_CONVENTIONS = ("inverted", "raw")
EXPECTED_N_TEST_IDENTITIES = 299
EXPECTED_N_GALLERY = 1998

DUPLICATE_PROBES = {("029_L", "3"), ("230_R", "5"), ("236_R", "2"), ("543_L", "2"), ("703_L", "7")}


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def available_probe_stems_2d(identity_dir: Path) -> list:
    return [s for s in PROBE_STEMS
            if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()]


def available_probe_stems_christina(identity_dir: Path) -> list:
    return [s for s in PROBE_STEMS
            if (identity_dir / f"{s}_code.npy").exists() and (identity_dir / f"{s}_mask.npy").exists()]


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
    return np.load(identity_dir / f"{stem}_code.npy"), np.load(identity_dir / f"{stem}_mask.npy")


def mask_for_convention(native_mask: np.ndarray, convention: str) -> np.ndarray:
    if convention == "inverted":
        return 1 - native_mask
    elif convention == "raw":
        return native_mask
    raise ValueError(convention)


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"{_OUT_PATH} already exists; write a new filename instead.")

    split = load_split(_SPLIT_PATH)
    train_ids, val_ids = split["identities"]["train"], split["identities"]["val"]
    test_ids = sorted(split["identities"]["test"])
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)
    assert len(test_ids) == EXPECTED_N_TEST_IDENTITIES
    assert len(full_gallery_ids) == EXPECTED_N_GALLERY

    nn_dedup = json.loads(_NN_DEDUP_PATH.read_text())
    nn_entry = {
        "dim": 64,
        "n_gallery": nn_dedup["quantized_deduplicated"]["n_gallery"],
        "n_probes": nn_dedup["quantized_deduplicated"]["n_probes"],
        "median_rank": nn_dedup["quantized_deduplicated"]["median_rank"],
        "mean_rank": nn_dedup["quantized_deduplicated"]["mean_rank"],
        **{f"recall_at_{k}": nn_dedup["quantized_deduplicated"][f"recall_at_{k}"] for k in K_VALUES},
    }

    print("\n--- Hand-designed features (casia-codes-2d), de-duplicated ---")
    t0 = time.time()
    gallery_ids = []
    gallery_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for identity in full_gallery_ids:
        d = _CASIA_2D_DIR / identity
        g_t, g_m = load_template(d, "1"), load_mask_2d(d, "1")
        feats = extract_features_maskaware(g_t, g_m)
        gallery_ids.append(identity)
        for name in HAND_FEATURE_NAMES:
            gallery_vecs[name].append(feats[name])

    probe_true_ids = []
    probe_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    n_skipped = 0
    for identity in test_ids:
        d = _CASIA_2D_DIR / identity
        for stem in available_probe_stems_2d(d):
            if (identity, stem) in DUPLICATE_PROBES:
                n_skipped += 1
                continue
            p_t, p_m = load_template(d, stem), load_mask_2d(d, stem)
            feats = extract_features_maskaware(p_t, p_m)
            probe_true_ids.append(identity)
            for name in HAND_FEATURE_NAMES:
                probe_vecs[name].append(feats[name])

    n_gallery, n_probe = len(gallery_ids), len(probe_true_ids)
    print(f"Extracted {n_gallery} galleries, {n_probe} probes (skipped {n_skipped} duplicates), "
          f"{time.time()-t0:.1f}s.")
    assert n_skipped == 5

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
            "dim": int(gallery.shape[1]), "n_gallery": n_gallery, "n_probes": n_probe,
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
            "eer": eer(genuine, impostor), "d_prime": d_prime(genuine, impostor),
        }
        for k in K_VALUES:
            entry[f"recall_at_{k}"] = recall_at_k(ranks, k)
        hand_results[name] = entry
        print(f"  {name}: R@10={entry['recall_at_10']:.4f} R@50={entry['recall_at_50']:.4f} "
              f"R@100={entry['recall_at_100']:.4f} R@300={entry['recall_at_300']:.4f} "
              f"median={entry['median_rank']:.1f} mean={entry['mean_rank']:.2f}")

    print("\n--- Christina's feature (casia-codes-christina), de-duplicated ---")
    config = FilterFHEConfig()
    christina_results = {}
    for convention in CHRISTINA_CONVENTIONS:
        print(f"\nConvention: {convention}")
        t0 = time.time()
        gallery_feats = {}
        for identity in full_gallery_ids:
            d = _CASIA_CHRISTINA_DIR / identity
            gcode, gmask_native = load_christina_pair(d, "1")
            gallery_feats[identity] = compute_template_features(gcode, mask_for_convention(gmask_native, convention))
        c_gallery_ids = list(gallery_feats.keys())

        probe_records = []
        skipped = 0
        for identity in test_ids:
            d = _CASIA_CHRISTINA_DIR / identity
            for stem in available_probe_stems_christina(d):
                if (identity, stem) in DUPLICATE_PROBES:
                    skipped += 1
                    continue
                pcode, pmask_native = load_christina_pair(d, stem)
                pfeat = compute_template_features(pcode, mask_for_convention(pmask_native, convention))
                probe_records.append((identity, pfeat))
        assert skipped == 5
        dim = len(extract_feature_vector_all_scales(next(iter(gallery_feats.values()))))
        print(f"  {len(c_gallery_ids)} galleries, {len(probe_records)} probes (skipped {skipped}), "
              f"dim={dim} (extraction {time.time()-t0:.1f}s)")

        t0 = time.time()
        ranks = np.empty(len(probe_records), dtype=np.int64)
        for i, (true_identity, pfeat) in enumerate(probe_records):
            scores = [(gid, compute_l1_distance_plaintext(pfeat, gallery_feats[gid], config)) for gid in c_gallery_ids]
            scores.sort(key=lambda x: x[1])
            ranked_ids = [gid for gid, _ in scores]
            ranks[i] = ranked_ids.index(true_identity) + 1
            if (i + 1) % 500 == 0:
                print(f"    ranked {i+1}/{len(probe_records)} ({time.time()-t0:.1f}s elapsed)...")

        entry = {
            "convention": convention, "dim": dim, "n_gallery": len(c_gallery_ids), "n_probes": len(probe_records),
            "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
        }
        for k in K_VALUES:
            entry[f"recall_at_{k}"] = recall_at_k(ranks, k)
        christina_results[convention] = entry
        print(f"  {convention}: R@10={entry['recall_at_10']:.4f} R@50={entry['recall_at_50']:.4f} "
              f"R@100={entry['recall_at_100']:.4f} R@300={entry['recall_at_300']:.4f} "
              f"median={entry['median_rank']:.1f} mean={entry['mean_rank']:.2f} "
              f"(ranking {time.time()-t0:.1f}s)")

    combined = dict(hand_results)
    combined["Christina (inverted)"] = christina_results["inverted"]
    combined["Christina (raw)"] = christina_results["raw"]
    combined["NN (v3, learned, quantized)"] = nn_entry

    print("\n" + "=" * 118)
    print(f"{'Feature':<28}{'Dim':>6}" + "".join(f"{'R@'+str(k):>9}" for k in K_VALUES) + f"{'MedRank':>10}{'MeanRank':>10}")
    print("-" * 118)
    for name, r in combined.items():
        print(f"{name:<28}{r['dim']:>6}" + "".join(f"{r[f'recall_at_{k}']:>9.4f}" for k in K_VALUES) +
              f"{r['median_rank']:>10.1f}{r['mean_rank']:>10.2f}")
    print("=" * 118)

    out = {
        "metadata": {
            "purpose": "De-duplicated re-run of all_features_nn_testsplit_comparison.json: excludes the "
                       "5 test probes byte-identical to their own gallery image (CASIA-Iris-Thousand "
                       "source-data artifact) from every feature's evaluation, not just the NN.",
            "duplicate_probes_excluded": sorted(f"{a}_{b}" for a, b in DUPLICATE_PROBES),
            "split_path": str(_SPLIT_PATH.relative_to(_ROOT)),
            "n_test_identities": len(test_ids),
            "n_gallery_identities": n_gallery,
            "n_probes_hand_designed": n_probe,
            "reference_undeduplicated_file": "level1-features/results/all_features_nn_testsplit_comparison.json",
            "nn_source": "level1-features/results/audit_nn_v3_dedup_metrics.json (quantized_deduplicated)",
            "distance_metrics": {"DFT/AC/RL-C9/RL-C6/NN": "SSD", "Christina": "L1 (compute_l1_distance_plaintext)"},
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "results": combined,
    }
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
