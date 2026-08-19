#!/usr/bin/env python3
"""Mask-blind counterpart to eval_myfeatures_realcasia_full2000.py (which is mask-aware
only), so the report's mask-handling section can compare blind-vs-aware at the same
full 2000-identity scale rather than only at the earlier N=300 scale.

Identical data, gallery/probe handling, and SSD-ranking method as
eval_myfeatures_realcasia_full2000.py: casia-extraction/casia-codes-2d/, 1998 usable
galleries (524_L/668_L excluded for a failed gallery image, per manifest.json's
gallery_ok flag; their probes are excluded with them since a probe with no gallery to
match against cannot yield a genuine comparison), 16,457 probes.

DFT and RL-C9/RL-C6 are called with mask=None (the default), which is each feature's
literal mask-blind path (see dft_spectrum.code_magnitude_spectrum / rl_features.
extract_rl_vector docstrings: "Default None reproduces the exact mask-blind behavior
used for all existing synthetic (SIC-Gen) results"). AC (ac_features.extract_ac_vector)
has no meaningfully different mask-blind mode -- row_autocorrelation is inherently
mask-aware by construction (it only counts bit-pairs where both positions are valid)
-- so AC is not re-run here; its number is identical to the one already reported in
myfeatures_realcasia_full2000.json.
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "myfeatures_realcasia_full2000_maskblind.json"
_MASKAWARE_PATH = _RESULTS / "myfeatures_realcasia_full2000.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from dft_spectrum import code_magnitude_spectrum, load_template  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)
FEATURE_NAMES = ("DFT", "RL-C9", "RL-C6")
K_VALUES = [10, 20, 50, 100, 150, 300]
PROBE_STEMS = [str(i) for i in range(2, 11)]
EXPECTED_N_GALLERY = 1998

MASK_HANDLING = {
    "DFT": "mask-blind -- code_magnitude_spectrum(template, mask=None) (default). Every row "
           "is FFT'd as-is, occluded bits included, exactly as for all synthetic (SIC-Gen) "
           "results.",
    "RL-C9": "mask-blind -- extract_rl_vector(template, C=9, mask=None) (default). Run-lengths "
              "computed over the raw bit sequence with no occlusion handling.",
    "RL-C6": "mask-blind -- same extract_rl_vector(..., mask=None) path as RL-C9, just C=6.",
}


def load_manifest():
    return json.loads(_CASIA_MANIFEST.read_text())


def available_probe_stems(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()
    ]


def extract_features_maskblind(template: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=None)[DFT_BINS],
        "RL-C9": extract_rl_vector(template, C=9, mask=None).astype(np.float64),
        "RL-C6": extract_rl_vector(template, C=6, mask=None).astype(np.float64),
    }


def dense_ranks(D: np.ndarray) -> np.ndarray:
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

    manifest = load_manifest()
    identities = manifest["identities"]
    n_identities_total = len(identities)
    excluded = [ident["identity"] for ident in identities if not ident["gallery_ok"]]
    print(f"Loaded manifest: {n_identities_total} identities total")
    print(f"Excluded (failed gallery, image 00): {excluded}")

    print("\nMask handling per feature:")
    for name in FEATURE_NAMES:
        print(f"  {name}: {MASK_HANDLING[name]}")

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
        g_feats = extract_features_maskblind(g_template)
        gallery_ids.append(identity)
        for name in FEATURE_NAMES:
            gallery_vecs[name].append(g_feats[name])

        for stem in available_probe_stems(identity_dir):
            p_template = load_template(identity_dir, stem)
            p_feats = extract_features_maskblind(p_template)
            probe_true_ids.append(identity)
            for name in FEATURE_NAMES:
                probe_vecs[name].append(p_feats[name])

    n_gallery = len(gallery_ids)
    n_probe = len(probe_true_ids)
    print(f"\nExtracted {n_gallery} galleries, {n_probe} probes.")
    if n_gallery != EXPECTED_N_GALLERY:
        print(f"WARNING: expected {EXPECTED_N_GALLERY} usable galleries, got {n_gallery}.")

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(n_probe)

    results = {}
    for name in FEATURE_NAMES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery, probe)

        rank_matrix = dense_ranks(D)
        ranks = rank_matrix[row_idx, true_col] + 1

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
              f"R@50={entry['recall@50']:.3f}, R@100={entry['recall@100']:.3f}, "
              f"median_rank={entry['median_rank']:.1f}")

    # ---- Blind-vs-aware comparison table (loads mask-aware numbers, does not recompute) ----
    maskaware = json.loads(_MASKAWARE_PATH.read_text())["results_real_full2000"]
    print("\n" + "=" * 110)
    print(f"{'Feature':<8}{'Convention':<12}{'Dim':>6}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'MedRank':>9}")
    print("-" * 110)
    for name in FEATURE_NAMES:
        for label, r in (("mask-blind", results[name]), ("mask-aware", maskaware[name])):
            print(f"{name:<8}{label:<12}{r['dim']:>6}{r['recall@10']:>8.3f}{r['recall@50']:>8.3f}"
                  f"{r['recall@100']:>8.3f}{r['median_rank']:>9.1f}")
    print(f"{'AC':<8}{'unchanged':<12}"
          f"{'AC is inherently mask-aware (row_autocorrelation only counts bit-pairs where both '}"
          f"{'positions are valid); no distinct mask-blind variant. See myfeatures_realcasia_full2000.json.'}")
    print("=" * 110)

    out = {
        "metadata": {
            "data_source_real": "casia-extraction/casia-codes-2d/ (real CASIA-Iris-Thousand via Open Iris)",
            "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
            "subject_selection_rule": manifest["subject_selection_rule"],
            "real_identity_count_total": n_identities_total,
            "real_gallery_size": n_gallery,
            "expected_gallery_size": EXPECTED_N_GALLERY,
            "real_probe_count": n_probe,
            "excluded_identities": excluded,
            "exclusion_reason": "gallery image (index 00) failed segmentation (gallery_ok=False in "
                                 "manifest.json); the identity and all its probes are excluded, not "
                                 "just the gallery slot, since a probe with no gallery to match "
                                 "against cannot yield a genuine comparison.",
            "mask_convention": "mask-blind (mask=None default path for DFT and RL-C9/RL-C6)",
            "mask_handling": MASK_HANDLING,
            "ac_note": "AC not re-run: extract_ac_vector is inherently mask-aware by construction "
                       "(row_autocorrelation only counts bit-pairs where both positions are valid), "
                       "so it has no meaningfully different mask-blind variant. Its number is "
                       "identical to the one already reported in myfeatures_realcasia_full2000.json.",
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "k_values": K_VALUES,
            "method_note": "Same feature extraction (mask=None), SSD ranking (ssd_matrix/eer/d_prime "
                            "imported unmodified from stage2_evaluate.py) and gallery/probe handling "
                            "as eval_myfeatures_realcasia_full2000.py, for direct blind-vs-aware "
                            "comparison against that file's results_real_full2000.",
            "maskaware_counterpart": "level1-features/results/myfeatures_realcasia_full2000.json",
        },
        "results_real_full2000_maskblind": results,
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
