#!/usr/bin/env python3
"""Real-CASIA validation of Stage 2's synthetic head-to-head (stage2_evaluate.py).

Runs the same four Level-1 features (DFT, AC, RL-C9, RL-C6), with the same
extraction calls, aggregation choices, and SSD-ranking metric as stage2_evaluate.py,
on real CASIA-Iris-Thousand codes (casia-extraction/casia-codes-2d/) instead of
synthetic SIC-Gen codes.

The one structural difference from stage2_evaluate.py: that script assumes exactly
one gallery (IC1) and one probe (IC2) per subject, so it can read off each probe's
true-match rank from the diagonal of a square SSD matrix. Real CASIA identities have
1 gallery (image 00) but 3-9 probes each (images 01-09, minus Open Iris failures --
see casia-extraction/manifest.json), so gallery and probe counts differ (300 vs
2441) and there is no positional correspondence. This script generalizes the ranking
step accordingly: for every probe, SSD is computed against all 300 galleries (via
stage2_evaluate.py's own ssd_matrix(), imported unmodified), and the true gallery's
rank is looked up by identity rather than assumed to sit on the diagonal. Recall@K,
median/mean rank, EER, d-prime, and the Spearman/Jaccard error-correlation analysis
are otherwise computed exactly as in stage2_evaluate.py (eer() and d_prime() are
imported and reused unmodified; the dense-rank tie-break -- double argsort -- is the
same one stage2_evaluate.py uses).

Mask handling (checked by reading the feature modules, not assumed -- see
"mask_handling" in the output metadata and printed below): DFT and RL ignore the
mask entirely (no mask parameter in their extraction functions). AC does use the
mask (row_autocorrelation only counts bit-pairs where both positions are valid).
No feature logic is changed here; all four are run exactly as stage2_evaluate.py
calls them.
"""

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
_OUT_PATH = _RESULTS / "myfeatures_realcasia.json"
_STAGE2_RESULTS_PATH = _RESULTS / "stage2_results.json"
_STAGE2_RANKS_PATH = _RESULTS / "stage2_ranks.npz"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix, eer, d_prime  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)  # bins 30-50 inclusive, matching stage2_evaluate.py
FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
K_VALUES = [10, 20, 50, 100, 150, 300]
FAILURE_RANK_THRESHOLD = 50
PROBE_STEMS = [str(i) for i in range(2, 11)]

MASK_HANDLING = {
    "DFT": "mask-blind -- code_magnitude_spectrum(template, ...) takes only the template, "
           "no mask parameter anywhere in dft_spectrum.py. On real CASIA (67.7% valid), "
           "occluded bits are included in the FFT as if they were real iris texture; "
           "this was built and tuned on synthetic SIC-Gen codes, which are ~90-100% valid, "
           "so this caveat had little effect there but matters here.",
    "AC": "mask-aware -- row_autocorrelation(row, mask) computes joint_valid = m * m_shifted "
          "and divides by its sum, so only bit-pairs where BOTH positions are valid (mask==1) "
          "contribute to R(lag); occluded bits are correctly excluded, using the same 1=valid "
          "convention real CASIA masks use. No caveat needed for AC.",
    "RL-C9": "mask-blind -- extract_rl_vector(template, C) takes only the template, no mask "
             "parameter in rl_features.py. Run-length histograms are computed over the raw "
             "bit array including occluded positions, which can create spurious/split runs "
             "at occlusion boundaries on real data.",
    "RL-C6": "mask-blind -- same extract_rl_vector() as RL-C9, just C=6; identical caveat.",
}

MASK_HANDLING_NOT_CHANGED_NOTE = (
    "No feature logic was modified for this run. DFT and RL are evaluated mask-blind, "
    "exactly as stage2_evaluate.py calls them, for comparability with the synthetic Stage 2 "
    "numbers; this means real occlusion is not being excluded for those two features, which "
    "the recall/EER numbers below should be read in light of."
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


def extract_features(template: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average")[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9).astype(np.float64),
        "RL-C6": extract_rl_vector(template, C=6).astype(np.float64),
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

    manifest = load_manifest()
    identities = manifest["identities"]
    n_identities = len(identities)
    print(f"Loaded manifest: {n_identities} identities")
    print(f"Subject selection: {manifest['subject_selection_rule']}")

    print("\nMask handling per feature (checked by reading the code, not assumed):")
    for name in FEATURE_NAMES:
        print(f"  {name}: {MASK_HANDLING[name]}")

    # ---- Extraction ----
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
        g_feats = extract_features(g_template, g_mask)
        gallery_ids.append(identity)
        for name in FEATURE_NAMES:
            gallery_vecs[name].append(g_feats[name])

        for stem in available_probe_stems(identity_dir):
            p_template = load_template(identity_dir, stem)
            p_mask = load_mask(identity_dir, stem)
            p_feats = extract_features(p_template, p_mask)
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

    # ---- Error-correlation analysis (real data) ----
    import itertools
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

    print("\nSpearman rank correlation (real CASIA):")
    for pair, v in spearman_results.items():
        print(f"  {pair}: rho={v['rho']:.4f} (p={v['pval']:.2e})")
    print(f"\nFailure-set Jaccard overlap (rank > {FAILURE_RANK_THRESHOLD}), real CASIA:")
    for pair, jac in jaccard_results.items():
        print(f"  {pair}: {jac:.3f}")

    # ---- Synthetic Stage 2 comparison (2000-subject SIC-Gen gallery) ----
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

    # ---- Combined side-by-side table ----
    print("\n" + "=" * 130)
    print(f"{'Feature':<8}{'Dataset':<10}{'Dim':>6}" +
          "".join(f"{'R@'+str(k):>8}" for k in K_VALUES) +
          f"{'MedRank':>9}{'MeanRank':>10}{'EER':>8}{'d-prime':>9}")
    print("-" * 130)
    for name in FEATURE_NAMES:
        for dataset, table in (("real(N=300)", results), ("synth(N=2000)", synthetic_table)):
            r = table[name]
            print(f"{name:<8}{dataset:<10}{r['dim']:>6}" +
                  "".join(f"{r[f'recall@{k}']:>8.3f}" for k in K_VALUES) +
                  f"{r['median_rank']:>9.1f}{r['mean_rank']:>10.1f}{r['eer']:>8.4f}{r['d_prime']:>9.3f}")
    print("=" * 130)

    out = {
        "metadata": {
            "data_source_real": "casia-extraction/casia-codes-2d/ (real CASIA-Iris-Thousand via Open Iris)",
            "data_source_synthetic": "level1-features/results/stage2_results.json + stage2_ranks.npz "
                                      "(SIC-Gen sicgen_gallery_2000subjects -- NOTE: 2000 subjects, not 300; "
                                      "recall@K is not directly comparable across different gallery sizes N, "
                                      "since more distractor candidates generally lowers recall at fixed K "
                                      "independent of feature quality)",
            "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
            "real_identity_count": n_identities,
            "real_gallery_size": n_gallery,
            "real_probe_count": n_probe,
            "subject_selection_rule": manifest["subject_selection_rule"],
            "synthetic_gallery_size": stage2["n_subjects"],
            "mask_handling": MASK_HANDLING,
            "mask_handling_note": MASK_HANDLING_NOT_CHANGED_NOTE,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "k_values": K_VALUES,
            "failure_rank_threshold": FAILURE_RANK_THRESHOLD,
            "method_note": (
                "Same feature extraction and SSD-ranking method as stage2_evaluate.py "
                "(ssd_matrix, eer, d_prime imported unmodified). Ranking generalized from "
                "stage2_evaluate.py's diagonal-of-square-matrix assumption (1 probe per subject, "
                "same order as gallery) to real CASIA's 1 gallery + 3-9 probes per identity: "
                "every probe is scored against all 300 galleries, and its true gallery's rank is "
                "looked up by identity rather than assumed to be at the same row/column index."
            ),
        },
        "results_real": results,
        "results_synthetic": synthetic_table,
        "spearman_real": spearman_results,
        "jaccard_failure_overlap_real": jaccard_results,
        "failure_set_sizes_real": {k: len(v) for k, v in failure_sets.items()},
        "spearman_synthetic": stage2["spearman"],
        "jaccard_failure_overlap_synthetic": stage2["jaccard_failure_overlap"],
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
