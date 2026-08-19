#!/usr/bin/env python3
"""RRF fusion (DFT+AC, DFT+RL-C9, AC+RL-C9, DFT+AC+RL-C9) at full 2000-identity real-CASIA
scale, removing the gallery-size confound from the earlier 300-identity fusion run
(fusion_maskaware_synth_real.json, real n_gallery=300).

Real-side extraction/fusion mechanics are identical to eval_fusion_maskaware.py's run_real()
(mask-aware DFT/AC/RL-C9, RRF k=60, full per-candidate dense-rank matrices, true rank looked
up by identity via the manifest), just run over the full 1000-subject manifest. The two
identities with a failed gallery image (524_L, 668_L) are excluded exactly as run_real()
already excludes any gallery_ok=False identity (skipped before its probes are ever added),
leaving 1998 usable galleries.

The synthetic side is NOT recomputed -- fusion_maskaware_synth_real.json's run_synthetic()
was already at full N=2000 scale (sic-gen/sicgen_gallery_2000subjects/), so those numbers
are loaded as-is from that file for direct comparison.
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
_OUT_PATH = _RESULTS / "fusion_realcasia_full2000.json"
_SYNTH_FUSION_PATH = _RESULTS / "fusion_maskaware_synth_real.json"

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
EXPECTED_N_GALLERY = 1998


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


def load_mask(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def available_probe_stems(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_template.txt").exists() and (identity_dir / f"{s}_mask.txt").exists()
    ]


def extract_features(template: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
    }


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


def run_real():
    print(f"\n{'='*20} REAL CASIA, full 2000 ({_CASIA_2D_DIR.name}) {'='*20}")
    manifest = json.loads(_CASIA_MANIFEST.read_text())
    identities = manifest["identities"]
    excluded = [ident["identity"] for ident in identities if not ident["gallery_ok"]]
    print(f"Excluded (failed gallery): {excluded}")

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
    if n_gallery != EXPECTED_N_GALLERY:
        print(f"WARNING: expected {EXPECTED_N_GALLERY} usable galleries, got {n_gallery}.")

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
        R = np.argsort(np.argsort(D, axis=1), axis=1) + 1
        rank_matrices[name] = R
        true_ranks[name] = R[row_idx, true_col]
        dims[name] = gallery.shape[1]

    rows = {name: metrics_row(true_ranks[name], dims[name]) for name in FUSION_FEATURES}

    for combo_name, feats in COMBOS.items():
        S = rrf_score_matrix([rank_matrices[f] for f in feats])
        fused_R = np.argsort(np.argsort(-S, axis=1), axis=1) + 1
        fused_true = fused_R[row_idx, true_col]
        rows[combo_name] = metrics_row(fused_true, sum(dims[f] for f in feats))

    spearman_results, jaccard_results = error_correlation(true_ranks)
    return {
        "n_gallery": n_gallery, "n_probe": n_probe, "rows": rows,
        "spearman": spearman_results, "jaccard": jaccard_results,
        "subject_selection_rule": manifest["subject_selection_rule"],
        "excluded_identities": excluded,
    }


def print_table(title: str, rows: dict):
    print(f"\n{title}")
    print("=" * 78)
    print(f"{'Name':<16}{'R@10':>8}{'R@50':>8}{'R@100':>8}{'MedRank':>10}{'MeanRank':>10}")
    print("-" * 78)
    for name, m in rows.items():
        print(f"{name:<16}{m['recall@10']:>8.3f}{m['recall@50']:>8.3f}{m['recall@100']:>8.3f}"
              f"{m['median_rank']:>10.1f}{m['mean_rank']:>10.1f}")
    print("=" * 78)


def gains(rows: dict, metric: str) -> dict:
    g = {}
    for combo_name, feats in COMBOS.items():
        best_component = max(rows[f][metric] for f in feats)
        fused = rows[combo_name][metric]
        g[combo_name] = fused - best_component
    return g


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"{_OUT_PATH} already exists; write a new filename instead.")
    if not _SYNTH_FUSION_PATH.exists():
        raise FileNotFoundError(f"expected existing synthetic fusion results at {_SYNTH_FUSION_PATH}")

    synth_fusion = json.loads(_SYNTH_FUSION_PATH.read_text())
    synth_meta = synth_fusion["metadata"]["datasets"]["synthetic"]
    synth_rows = synth_fusion["results_synthetic"]
    synth_spearman = synth_fusion["spearman_synthetic_maskaware"]
    synth_jaccard = synth_fusion["jaccard_failure_overlap_synthetic_maskaware"]
    print(f"Loaded existing synthetic fusion results (N={synth_meta['n_subjects']}) from {_SYNTH_FUSION_PATH}")

    real = run_real()

    print_table(f"Synthetic (N={synth_meta['n_subjects']}), mask-aware RRF fusion [loaded]:", synth_rows)
    print_table(f"Real CASIA full2000 (N={real['n_gallery']} galleries / {real['n_probe']} probes), "
                "mask-aware RRF fusion:", real["rows"])

    synth_gains_r10 = gains(synth_rows, "recall@10")
    real_gains_r10 = gains(real["rows"], "recall@10")
    synth_gains_r50 = gains(synth_rows, "recall@50")
    real_gains_r50 = gains(real["rows"], "recall@50")

    print("\nR@10 fusion gain over the better single component (synthetic vs real-full2000):")
    print(f"{'Combo':<16}{'Synth gain':>12}{'Real gain':>12}")
    for combo_name in COMBOS:
        print(f"{combo_name:<16}{synth_gains_r10[combo_name]:>+12.3f}{real_gains_r10[combo_name]:>+12.3f}")

    dft_rl_beats_dft_ac = real["rows"]["DFT+RL-C9"]["recall@10"] > real["rows"]["DFT+AC"]["recall@10"]
    dft_rl_vs_dft_ac_note = (
        f"DFT+RL-C9 R@10={real['rows']['DFT+RL-C9']['recall@10']:.3f} vs "
        f"DFT+AC R@10={real['rows']['DFT+AC']['recall@10']:.3f} on real-full2000 -- "
        f"DFT+RL-C9 {'still beats' if dft_rl_beats_dft_ac else 'no longer beats'} DFT+AC."
    )
    print("\n" + dft_rl_vs_dft_ac_note)

    print("\nError correlation (mask-aware), synthetic vs real-full2000:")
    for pair in ("DFT-AC", "DFT-RL-C9", "AC-RL-C9"):
        s_rho = synth_spearman[pair]["rho"]
        r_rho = real["spearman"][pair]["rho"]
        s_jac = synth_jaccard[pair]
        r_jac = real["jaccard"][pair]
        print(f"  {pair}: spearman synth={s_rho:.4f} real={r_rho:.4f}; "
              f"jaccard synth={s_jac:.3f} real={r_jac:.3f}")

    out = {
        "metadata": {
            "datasets": {
                "synthetic": {**synth_meta, "source": "loaded as-is from fusion_maskaware_synth_real.json "
                                                        "(already N=2000, not recomputed)"},
                "real": {
                    "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
                    "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
                    "n_gallery": real["n_gallery"],
                    "expected_n_gallery": EXPECTED_N_GALLERY,
                    "n_probe": real["n_probe"],
                    "subject_selection_rule": real["subject_selection_rule"],
                    "excluded_identities": real["excluded_identities"],
                    "exclusion_reason": "gallery image (index 00) failed segmentation "
                                         "(gallery_ok=False); identity and its probes excluded entirely.",
                },
            },
            "features": list(FUSION_FEATURES),
            "mask_aware": True,
            "rrf_k": RRF_K,
            "fusion_combos": {k: list(v) for k, v in COMBOS.items()},
            "k_values": K_VALUES,
            "failure_rank_threshold": FAILURE_RANK_THRESHOLD,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "dft_rl_vs_dft_ac_note": dft_rl_vs_dft_ac_note,
            "prior_run_for_comparison": "level1-features/results/fusion_maskaware_synth_real.json "
                                         "(real N=300 galleries, first 150 subjects) -- not overwritten.",
            "method_note": "RRF: fused score(candidate) = sum over included features of 1/(k+rank), "
                            "rank = 1-indexed dense rank (double-argsort tie-break) per feature's SSD "
                            "matrix (ssd_matrix imported unmodified from stage2_evaluate.py). Real side "
                            "recomputed at full 1000-subject scale via the same non-bijective, "
                            "manifest-driven true-rank lookup as eval_fusion_maskaware.py's run_real(); "
                            "synthetic side reused unchanged since it was already N=2000.",
        },
        "results_synthetic": synth_rows,
        "results_real": real["rows"],
        "r10_fusion_gain_synthetic": synth_gains_r10,
        "r10_fusion_gain_real": real_gains_r10,
        "r50_fusion_gain_synthetic": synth_gains_r50,
        "r50_fusion_gain_real": real_gains_r50,
        "spearman_synthetic_maskaware": synth_spearman,
        "jaccard_failure_overlap_synthetic_maskaware": synth_jaccard,
        "spearman_real_maskaware": real["spearman"],
        "jaccard_failure_overlap_real_maskaware": real["jaccard"],
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
