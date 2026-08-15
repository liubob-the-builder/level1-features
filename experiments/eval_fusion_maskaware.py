#!/usr/bin/env python3
"""RRF fusion (DFT, AC, RL-C9) with mask-aware features, on both synthetic and real data.

Reuses the same mask-aware feature paths as myfeatures_synthetic_maskaware.json /
myfeatures_realcasia_maskaware.json (dft_spectrum.code_magnitude_spectrum(..., mask=mask),
rl_features.extract_rl_vector(..., mask=mask); AC unchanged, was already mask-aware) and
the same RRF mechanics as the original stage3_fusion.py (full per-candidate rank matrices,
fused score = sum over features of 1/(k + rank), k=60, re-ranked by descending fused score).

Two datasets, two different gallery/probe structures, same RRF logic applied to each:
  - Synthetic: sic-gen/sicgen_gallery_2000subjects/, bijective (1 gallery + 1 probe per
    subject, same order) -- true-identity rank read off the SSD/RRF matrix diagonal, exactly
    as stage2_evaluate.py / stage3_fusion.py do.
  - Real: casia-extraction/casia-codes-2d/, 300 galleries (image 00) x 2441 probes (images
    01-09, non-bijective, variable count per identity) -- true-identity rank looked up by
    identity via the manifest, exactly as eval_myfeatures_realcasia_maskaware.py does. The
    per-probe candidate rank matrix (dense rank, double-argsort tie-break) generalizes the
    "full rank matrix" RRF needs on this non-square structure the same way.

Does not modify or overwrite stage3_fusion.py's outputs, stage2_results.json,
myfeatures_synthetic_maskaware.json, or myfeatures_realcasia_maskaware.json -- all are
read-only inputs or left alone entirely (this script re-extracts features itself rather
than reusing saved rank arrays, since RRF needs the full per-candidate rank list, not just
each probe's true-identity rank).
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
_SICGEN_DIR = _ROOT / "sic-gen" / "sicgen_gallery_2000subjects"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "fusion_maskaware_synth_real.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix  # noqa: E402 -- imported unmodified

DFT_BINS = np.arange(30, 51)
FUSION_FEATURES = ("DFT", "AC", "RL-C9")
COMBOS = {
    "DFT+AC": ("DFT", "AC"),
    "DFT+RL-C9": ("DFT", "RL-C9"),
    "AC+RL-C9": ("AC", "RL-C9"),
    "DFT+AC+RL-C9": ("DFT", "AC", "RL-C9"),
}
RRF_K = 60
K_VALUES = [10, 20, 50, 100, 150, 300]
FAILURE_RANK_THRESHOLD = 50
PROBE_STEMS = [str(i) for i in range(2, 11)]


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def metrics_row(ranks: np.ndarray, dim: int) -> dict:
    entry = {"dim": dim, "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks))}
    for k in K_VALUES:
        entry[f"recall@{k}"] = recall_at_k(ranks, k)
    return entry


def rrf_score_matrix(rank_matrices, k: int = RRF_K) -> np.ndarray:
    total = None
    for R in rank_matrices:
        contrib = 1.0 / (k + R)
        total = contrib if total is None else total + contrib
    return total


def error_correlation(ranks_by_feature: dict) -> tuple:
    pairs = list(itertools.combinations(FUSION_FEATURES, 2))
    spearman_results = {}
    for a, b in pairs:
        rho, pval = spearmanr(ranks_by_feature[a], ranks_by_feature[b])
        spearman_results[f"{a}-{b}"] = {"rho": float(rho), "pval": float(pval)}
    failure_sets = {
        name: set(np.where(ranks_by_feature[name] > FAILURE_RANK_THRESHOLD)[0].tolist())
        for name in FUSION_FEATURES
    }
    jaccard_results = {}
    for a, b in pairs:
        sa, sb = failure_sets[a], failure_sets[b]
        union = sa | sb
        jaccard_results[f"{a}-{b}"] = len(sa & sb) / len(union) if union else float("nan")
    return spearman_results, jaccard_results


def run_synthetic():
    print(f"\n{'='*20} SYNTHETIC ({_SICGEN_DIR.name}) {'='*20}")
    subject_dirs = sorted(
        (d for d in _SICGEN_DIR.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects")

    def load_mask(subject_dir, stem):
        with open(subject_dir / f"{stem}_mask.txt") as f:
            return np.array([list(r.strip()) for r in f], dtype=np.uint8)

    def extract(stem):
        dft, ac, rl9 = [], [], []
        for sd in subject_dirs:
            template = load_template(sd, stem)
            mask = load_mask(sd, stem)
            dft.append(code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS])
            ac.append(extract_ac_vector(template, mask, mode="concat"))
            rl9.append(extract_rl_vector(template, C=9, mask=mask).astype(np.float64))
        return {"DFT": np.stack(dft), "AC": np.stack(ac), "RL-C9": np.stack(rl9)}

    print("Extracting mask-aware gallery (IC1) and probe (IC2) features ...")
    gallery_feats = extract("1")
    probe_feats = extract("2")

    rank_matrices = {}
    true_ranks = {}
    dims = {}
    for name in FUSION_FEATURES:
        D = ssd_matrix(gallery_feats[name], probe_feats[name])
        R = np.argsort(np.argsort(D, axis=1), axis=1) + 1  # 1-indexed candidate rank
        rank_matrices[name] = R
        true_ranks[name] = R[np.arange(n), np.arange(n)]
        dims[name] = gallery_feats[name].shape[1]

    rows = {name: metrics_row(true_ranks[name], dims[name]) for name in FUSION_FEATURES}

    fused_ranks = {}
    for combo_name, feats in COMBOS.items():
        S = rrf_score_matrix([rank_matrices[f] for f in feats])
        fused_R = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        fused_true = fused_R[np.arange(n), np.arange(n)]
        fused_ranks[combo_name] = fused_true
        rows[combo_name] = metrics_row(fused_true, sum(dims[f] for f in feats))

    spearman_results, jaccard_results = error_correlation(true_ranks)
    return {"n": n, "rows": rows, "spearman": spearman_results, "jaccard": jaccard_results}


def run_real():
    print(f"\n{'='*20} REAL CASIA ({_CASIA_2D_DIR.name}) {'='*20}")
    manifest = json.loads(_CASIA_MANIFEST.read_text())
    identities = manifest["identities"]

    def load_mask(subject_dir, stem):
        with open(subject_dir / f"{stem}_mask.txt") as f:
            return np.array([list(r.strip()) for r in f], dtype=np.uint8)

    def available_probe_stems(identity_dir):
        return [
            s for s in PROBE_STEMS
            if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()
        ]

    def extract_features(template, mask):
        return {
            "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
            "AC": extract_ac_vector(template, mask, mode="concat"),
            "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
        }

    gallery_ids = []
    gallery_vecs = {name: [] for name in FUSION_FEATURES}
    probe_true_ids = []
    probe_vecs = {name: [] for name in FUSION_FEATURES}

    print("Extracting mask-aware gallery + probe features ...")
    for ident in identities:
        identity = ident["identity"]
        if not ident["gallery_ok"]:
            continue
        identity_dir = _CASIA_2D_DIR / identity

        g_template = load_template(identity_dir, "1")
        g_mask = load_mask(identity_dir, "1")
        g_feats = extract_features(g_template, g_mask)
        gallery_ids.append(identity)
        for name in FUSION_FEATURES:
            gallery_vecs[name].append(g_feats[name])

        for stem in available_probe_stems(identity_dir):
            p_template = load_template(identity_dir, stem)
            p_mask = load_mask(identity_dir, stem)
            p_feats = extract_features(p_template, p_mask)
            probe_true_ids.append(identity)
            for name in FUSION_FEATURES:
                probe_vecs[name].append(p_feats[name])

    n_gallery = len(gallery_ids)
    n_probe = len(probe_true_ids)
    print(f"Extracted {n_gallery} galleries, {n_probe} probes.")

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(n_probe)

    rank_matrices = {}
    true_ranks = {}
    dims = {}
    for name in FUSION_FEATURES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery, probe)
        R = np.argsort(np.argsort(D, axis=1), axis=1) + 1  # 1-indexed dense rank per candidate
        rank_matrices[name] = R
        true_ranks[name] = R[row_idx, true_col]
        dims[name] = gallery.shape[1]

    rows = {name: metrics_row(true_ranks[name], dims[name]) for name in FUSION_FEATURES}

    fused_ranks = {}
    for combo_name, feats in COMBOS.items():
        S = rrf_score_matrix([rank_matrices[f] for f in feats])
        fused_R = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        fused_true = fused_R[row_idx, true_col]
        fused_ranks[combo_name] = fused_true
        rows[combo_name] = metrics_row(fused_true, sum(dims[f] for f in feats))

    spearman_results, jaccard_results = error_correlation(true_ranks)
    return {"n_gallery": n_gallery, "n_probe": n_probe, "rows": rows,
            "spearman": spearman_results, "jaccard": jaccard_results,
            "subject_selection_rule": manifest["subject_selection_rule"]}


def print_table(title: str, rows: dict):
    print(f"\n{title}")
    print("=" * 78)
    print(f"{'Name':<16}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'MedRank':>10}{'MeanRank':>10}")
    print("-" * 78)
    for name, m in rows.items():
        print(f"{name:<16}{m['recall@10']:>8.3f}{m['recall@50']:>8.3f}{m['recall@100']:>8.3f}"
              f"{m['median_rank']:>10.1f}{m['mean_rank']:>10.1f}")
    print("=" * 78)


def gains(rows: dict) -> dict:
    g = {}
    for combo_name, feats in COMBOS.items():
        best_component_r50 = max(rows[f]["recall@50"] for f in feats)
        fused_r50 = rows[combo_name]["recall@50"]
        g[combo_name] = fused_r50 - best_component_r50
    return g


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"{_OUT_PATH} already exists; write a new filename instead.")

    synth = run_synthetic()
    real = run_real()

    print_table("Synthetic (N=2000), mask-aware RRF fusion:", synth["rows"])
    print_table("Real CASIA (N=300 galleries / 2441 probes), mask-aware RRF fusion:", real["rows"])

    synth_gains = gains(synth["rows"])
    real_gains = gains(real["rows"])

    print("\nR@50 fusion gain over the better single component (synthetic vs real):")
    print(f"{'Combo':<16}{'Synth gain':>12}{'Real gain':>12}")
    for combo_name in COMBOS:
        print(f"{combo_name:<16}{synth_gains[combo_name]:>+12.3f}{real_gains[combo_name]:>+12.3f}")

    best_synth_fusion = max(synth_gains, key=synth_gains.get)
    best_real_fusion = max(real_gains, key=real_gains.get)
    best_synth_single_r50 = max(synth["rows"][f]["recall@50"] for f in FUSION_FEATURES)
    best_real_single_r50 = max(real["rows"][f]["recall@50"] for f in FUSION_FEATURES)
    print(f"\nBest fusion, synthetic: {best_synth_fusion} "
          f"(R@50={synth['rows'][best_synth_fusion]['recall@50']:.3f}, "
          f"best single R@50={best_synth_single_r50:.3f}, gain={synth_gains[best_synth_fusion]:+.3f})")
    print(f"Best fusion, real:      {best_real_fusion} "
          f"(R@50={real['rows'][best_real_fusion]['recall@50']:.3f}, "
          f"best single R@50={best_real_single_r50:.3f}, gain={real_gains[best_real_fusion]:+.3f})")

    print("\nError correlation (mask-aware), synthetic vs real:")
    for pair in ("DFT-AC", "DFT-RL-C9", "AC-RL-C9"):
        s_rho = synth["spearman"][pair]["rho"]
        r_rho = real["spearman"][pair]["rho"]
        s_jac = synth["jaccard"][pair]
        r_jac = real["jaccard"][pair]
        print(f"  {pair}: spearman synth={s_rho:.4f} real={r_rho:.4f}; "
              f"jaccard synth={s_jac:.3f} real={r_jac:.3f}")

    out = {
        "metadata": {
            "datasets": {
                "synthetic": {"gallery": str(_SICGEN_DIR.relative_to(_ROOT)), "n_subjects": synth["n"]},
                "real": {
                    "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
                    "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
                    "n_gallery": real["n_gallery"],
                    "n_probe": real["n_probe"],
                    "subject_selection_rule": real["subject_selection_rule"],
                },
            },
            "features": list(FUSION_FEATURES),
            "mask_aware": True,
            "mask_aware_note": "DFT and RL-C9 use their optional mask parameter (dft_spectrum.py / "
                                "rl_features.py); AC unchanged, was already mask-aware. Same mask-aware "
                                "extraction code as myfeatures_synthetic_maskaware.json / "
                                "myfeatures_realcasia_maskaware.json.",
            "rrf_k": RRF_K,
            "fusion_combos": {k: list(v) for k, v in COMBOS.items()},
            "k_values": K_VALUES,
            "failure_rank_threshold": FAILURE_RANK_THRESHOLD,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "method_note": "RRF: fused score(candidate) = sum over included features of 1/(k+rank), "
                            "rank = 1-indexed dense rank (double-argsort tie-break) per feature's SSD "
                            "matrix (ssd_matrix imported unmodified from stage2_evaluate.py). Synthetic "
                            "is bijective (true rank read off the diagonal, as in stage3_fusion.py); real "
                            "CASIA is non-bijective (true rank looked up by identity via the manifest, as "
                            "in eval_myfeatures_realcasia_maskaware.py). Does not overwrite stage3_fusion.py's "
                            "outputs or either dataset's existing mask-aware single-feature results files.",
        },
        "results_synthetic": synth["rows"],
        "results_real": real["rows"],
        "r50_fusion_gain_synthetic": synth_gains,
        "r50_fusion_gain_real": real_gains,
        "spearman_synthetic_maskaware": synth["spearman"],
        "jaccard_failure_overlap_synthetic_maskaware": synth["jaccard"],
        "spearman_real_maskaware": real["spearman"],
        "jaccard_failure_overlap_real_maskaware": real["jaccard"],
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
