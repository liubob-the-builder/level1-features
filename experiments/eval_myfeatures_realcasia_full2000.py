#!/usr/bin/env python3
"""Mask-aware real-CASIA evaluation at full 2000-identity scale (1000 subjects x 2 eyes),
removing the gallery-size confound from the earlier 300-identity run
(myfeatures_realcasia_maskaware.json, N=300 galleries).

Identical feature extraction / SSD-ranking method as eval_myfeatures_realcasia_maskaware.py
(DFT/AC/RL-C9/RL-C6, all mask-aware), just run over the full casia-extraction/casia-codes-2d/
export (casia-extraction/manifest.json now covers all 1000 subjects). Two identities (524_L,
668_L) have a failed gallery image (index 00) per the manifest's gallery_ok flag; they and
their probes are excluded exactly as eval_myfeatures_realcasia_maskaware.py already excludes
any gallery_ok=False identity (the `if not ident["gallery_ok"]: continue` skips the identity
entirely before its probes are ever added), leaving 1998 usable galleries.

Synthetic comparison numbers are loaded, not recomputed, from the existing 2000-subject
synthetic mask-aware results: myfeatures_synthetic_maskaware.json (results_maskaware; DFT/
RL-C9/RL-C6 only -- AC was excluded there because it's already mask-aware, identical under
both conventions by construction) plus stage2_results.json/stage2_ranks.npz for AC's numbers
and to backfill recall@20/150 (that file's K_VALUES was [10,50,100,300]) for all four
features, exactly as eval_myfeatures_realcasia_maskaware.py did for its own synthetic column.
"""

import itertools
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "myfeatures_realcasia_full2000.json"
_SYNTH_MASKAWARE_PATH = _RESULTS / "myfeatures_synthetic_maskaware.json"
_STAGE2_RESULTS_PATH = _RESULTS / "stage2_results.json"
_STAGE2_RANKS_PATH = _RESULTS / "stage2_ranks.npz"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template, MIN_VALID_FRAC  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)
FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
K_VALUES = [10, 20, 50, 100, 150, 300]
FAILURE_RANK_THRESHOLD = 50
PROBE_STEMS = [str(i) for i in range(2, 11)]
EXPECTED_N_GALLERY = 1998

MASK_HANDLING = {
    "DFT": f"mask-aware -- code_magnitude_spectrum(template, mask=mask). Per row: if valid "
           f"fraction < {MIN_VALID_FRAC} the row is dropped from the across-row average; kept "
           "rows have occluded bits mean-filled with that row's own valid-bit mean before the "
           "FFT. Bins 30-50 kept.",
    "AC": "mask-aware -- extract_ac_vector (already mask-aware: row_autocorrelation only counts "
          "bit-pairs where both positions are valid).",
    "RL-C9": "mask-aware -- extract_rl_vector(template, C=9, mask=mask). Run-lengths computed "
             "only within maximal contiguous runs of valid bits; an occluded bit terminates the "
             "run it interrupts and is not itself counted.",
    "RL-C6": "mask-aware -- same extract_rl_vector(..., mask=mask) path as RL-C9, just C=6.",
}


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
    return np.argsort(np.argsort(D, axis=1), axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def build_synthetic_table() -> dict:
    """Directly-comparable N=2000 synthetic table: mask-aware DFT/RL-C9/RL-C6 loaded from
    myfeatures_synthetic_maskaware.json, AC (mask-aware == mask-blind for AC) plus recall@20/150
    for all four loaded/recomputed from stage2_results.json + stage2_ranks.npz."""
    synth_aware = json.loads(_SYNTH_MASKAWARE_PATH.read_text())
    stage2 = json.loads(_STAGE2_RESULTS_PATH.read_text())
    stage2_ranks_npz = np.load(_STAGE2_RANKS_PATH)

    table = {}
    for name in FEATURE_NAMES:
        npz_key = f"ranks_{name.replace('-', '_')}"
        stage2_ranks = stage2_ranks_npz[npz_key]
        if name == "AC":
            base = dict(stage2["results_table"]["AC"])
        else:
            base = dict(synth_aware["results_maskaware"][name])
        # backfill recall@20/150 (absent from both source files' K_VALUES) from stage2 ranks;
        # for AC this also supplies recall@10/50/100/300 consistently with the rest.
        for k in K_VALUES:
            base[f"recall@{k}"] = recall_at_k(stage2_ranks, k)
        table[name] = base
    return table


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
    print(f"Subject selection: {manifest['subject_selection_rule']}")
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
    if n_gallery != EXPECTED_N_GALLERY:
        print(f"WARNING: expected {EXPECTED_N_GALLERY} usable galleries, got {n_gallery}.")

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(n_probe)

    results = {}
    ranks_by_feature = {}

    for name in FEATURE_NAMES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery, probe)

        rank_matrix = dense_ranks(D)
        ranks = rank_matrix[row_idx, true_col] + 1
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

    # ---- Error-correlation analysis (real data, full 2000, mask-aware) ----
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
        jaccard_results[f"{a}-{b}"] = len(sa & sb) / len(union) if union else float("nan")

    print("\nSpearman rank correlation (real CASIA, full 2000, mask-aware):")
    for pair, v in spearman_results.items():
        print(f"  {pair}: rho={v['rho']:.4f} (p={v['pval']:.2e})")
    print(f"\nFailure-set Jaccard overlap (rank > {FAILURE_RANK_THRESHOLD}):")
    for pair, jac in jaccard_results.items():
        print(f"  {pair}: {jac:.3f}")

    # ---- Synthetic comparison table (N=2000, mask-aware) ----
    synthetic_table = build_synthetic_table()

    # ---- Feature-ranking-reversal check (R@50, real-full2000 vs synthetic) ----
    def rank_order(table):
        return sorted(FEATURE_NAMES, key=lambda n: -table[n]["recall@50"])

    real_order = rank_order(results)
    synth_order = rank_order(synthetic_table)
    ranking_note = (
        f"By R@50: synthetic order = {synth_order}; real-full2000 order = {real_order}. "
        f"Earlier real-300 order was reported as RL~=DFT>AC; synthetic order was DFT~=AC>>RL."
    )
    print("\n" + ranking_note)

    # ---- Combined table print ----
    print("\n" + "=" * 140)
    print(f"{'Feature':<8}{'Dataset':<12}{'Dim':>6}" +
          "".join(f"{'R@'+str(k):>8}" for k in K_VALUES) +
          f"{'MedRank':>9}{'MeanRank':>10}{'EER':>8}{'d-prime':>9}")
    print("-" * 140)
    for name in FEATURE_NAMES:
        for dataset, r in (("real-2000", results[name]), ("synth-2000", synthetic_table[name])):
            print(f"{name:<8}{dataset:<12}{r['dim']:>6}" +
                  "".join(f"{r[f'recall@{k}']:>8.3f}" for k in K_VALUES) +
                  f"{r['median_rank']:>9.1f}{r['mean_rank']:>10.1f}{r['eer']:>8.4f}{r['d_prime']:>9.3f}")
    print("=" * 140)

    out = {
        "metadata": {
            "data_source_real": "casia-extraction/casia-codes-2d/ (real CASIA-Iris-Thousand via Open Iris)",
            "data_source_synthetic": "level1-features/results/myfeatures_synthetic_maskaware.json "
                                      "(DFT/RL-C9/RL-C6, mask-aware) + stage2_results.json/"
                                      "stage2_ranks.npz (AC, and recall@20/150 backfill for all four)",
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
            "synthetic_gallery_size": 2000,
            "mask_handling": MASK_HANDLING,
            "dft_min_valid_frac_threshold": MIN_VALID_FRAC,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "k_values": K_VALUES,
            "failure_rank_threshold": FAILURE_RANK_THRESHOLD,
            "ranking_reversal_check": ranking_note,
            "method_note": "Same feature extraction, SSD ranking (ssd_matrix/eer/d_prime imported "
                            "unmodified from stage2_evaluate.py) and gallery/probe handling as "
                            "eval_myfeatures_realcasia_maskaware.py, run over the full 1000-subject "
                            "manifest instead of the first 150.",
            "prior_run_for_comparison": "level1-features/results/myfeatures_realcasia_maskaware.json "
                                         "(N=300 galleries, first 150 subjects) -- not overwritten.",
        },
        "results_real_full2000": results,
        "results_synthetic_2000": synthetic_table,
        "spearman_real_full2000": spearman_results,
        "jaccard_failure_overlap_real_full2000": jaccard_results,
        "failure_set_sizes_real_full2000": {k: len(v) for k, v in failure_sets.items()},
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
