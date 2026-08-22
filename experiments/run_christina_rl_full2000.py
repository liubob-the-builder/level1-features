#!/usr/bin/env python3
"""Run Christina's enhanced-RL Level-1 feature on the full SIC-Gen 2000-subject
gallery, under two mask conventions, extending run_christina_rl.py's 300-subject run
to full Stage-2 scale.

Christina's `compute_enhanced_run_stats` calls `non_mask_adjacent_valid(mask_bool)`,
which treats its input as True=OCCLUDED (it returns `(~m) & (~adjacent)` as the valid
set). SIC-Gen's own mask convention is the opposite: 1=valid, 0=occluded (confirmed by
christina-fhe-fis/diagnose_masks.py). This script quantifies the effect of that
mismatch at full gallery scale:
  Run "inverted": occluded_mask = 1 - sicgen_mask, fed to her code (matches
      non_mask_adjacent_valid's own True=occluded expectation).
  Run "raw": sicgen_mask fed as-is, 1=valid (what her pipeline's literal wiring does --
      no inversion occurs anywhere in her real extract_iris_code path).

Gallery/probe split: IC1 = gallery, IC2 = genuine probe, per subject, full 2000-subject
gallery (no sampling). Codes are single-scale, shaped (1, 32, 512), matching
compute_template_features's expected (n_scales, H, W) input -- same convention as
run_christina_rl.py's 300-subject run, so the two are directly comparable.

For each convention: rank every probe (2000) against the full gallery (2000) by
compute_l1_distance_plaintext (imported from christina-fhe-fis, unmodified) --
2000x2000 = 4,000,000 pairwise distances -- and feed the per-probe rankings into
compute_recall_at_k (imported from christina-fhe-fis/evaluation.py, unmodified) for
Recall@10/20/50/100/150/300, median/mean rank, feature dimension, and % of rows with
no valid segment >= min_segment_len=8 (rows that contribute nothing to the
histogram/row-stats features).
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

_LEVEL1 = Path(__file__).resolve().parent.parent
_ROOT = _LEVEL1.parent
_CHRISTINA = _ROOT / "christina-fhe-fis"
_GALLERY = _ROOT / "sic-gen" / "sicgen_gallery_2000subjects"
_GALLERY_METADATA = _GALLERY / "metadata.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "christina_rl_maskconvention_full2000.json"

sys.path.insert(0, str(_CHRISTINA))

import numpy as np  # noqa: E402

from filter_fhe_iris_complete import (  # noqa: E402
    FilterFHEConfig,
    compute_l1_distance_plaintext,
    compute_template_features,
    extract_feature_vector_all_scales,
)
from evaluation import compute_recall_at_k  # noqa: E402

MIN_SEGMENT_LEN = 8
K_VALUES = [10, 20, 50, 100, 150, 300]
CONVENTIONS = ("inverted", "raw")


def load_template(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_template.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def load_mask(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def mask_for_convention(sicgen_mask: np.ndarray, convention: str) -> np.ndarray:
    """sicgen_mask: SIC-Gen's native convention, 1=valid, 0=occluded."""
    if convention == "inverted":
        return 1 - sicgen_mask  # 1=occluded, matches non_mask_adjacent_valid's expectation
    elif convention == "raw":
        return sicgen_mask  # fed as-is, 1=valid
    raise ValueError(convention)


def count_dropped_rows(features: dict) -> tuple:
    """Rows in row_stats with mean_run_0==0 and mean_run_1==0: no segment >= MIN_SEGMENT_LEN
    contributed to that row (includes rows with zero valid bits at all)."""
    dropped = 0
    total = 0
    for scale_key, stats in features.items():
        for row in stats["row_stats"]:
            total += 1
            if row["mean_run_0"] == 0 and row["mean_run_1"] == 0:
                dropped += 1
    return dropped, total


def extract_all(subject_dirs, stem: str, convention: str):
    """Returns dict subject_id -> features (Christina's nested dict format), plus
    running dropped/total row counts for this convention."""
    feats = {}
    dropped_total = 0
    rows_total = 0
    n = len(subject_dirs)
    for i, sd in enumerate(subject_dirs):
        subject_id = sd.name
        template = load_template(sd, stem)  # (32, 512), values in {0,1}
        sicgen_mask = load_mask(sd, stem)  # (32, 512), 1=valid
        mask = mask_for_convention(sicgen_mask, convention)

        code = template[None, :, :]  # (1, 32, 512) single scale
        mask3 = mask[None, :, :]

        features = compute_template_features(code, mask3)
        feats[subject_id] = features

        d, t = count_dropped_rows(features)
        dropped_total += d
        rows_total += t

        if (i + 1) % 500 == 0:
            print(f"    extracted {i + 1}/{n} ({stem}_*)...")

    return feats, dropped_total, rows_total


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_convention(subject_dirs, config: FilterFHEConfig, convention: str) -> dict:
    print(f"\n=== Convention: {convention} ===")
    print("Extracting gallery (IC1) features ...")
    gallery_feats, g_dropped, g_total = extract_all(subject_dirs, "1", convention)
    print("Extracting probe (IC2) features ...")
    probe_feats, p_dropped, p_total = extract_all(subject_dirs, "2", convention)

    subject_ids = [sd.name for sd in subject_dirs]
    gallery_ids = subject_ids  # IC1 identities, order fixed

    dim = len(extract_feature_vector_all_scales(gallery_feats[subject_ids[0]]))
    print(f"  {len(gallery_ids)} galleries, {len(subject_ids)} probes, feature dim={dim}")

    results = []
    n = len(subject_ids)
    for i, probe_id in enumerate(subject_ids):
        probe_features = probe_feats[probe_id]
        level1_scores = [
            (gid, compute_l1_distance_plaintext(probe_features, gallery_feats[gid], config))
            for gid in gallery_ids
        ]
        level1_scores.sort(key=lambda x: x[1])
        results.append({"true_identity": probe_id, "level1_scores": level1_scores})
        if (i + 1) % 200 == 0:
            print(f"    ranked {i + 1}/{n} probes...")

    recall_stats = compute_recall_at_k(results, k_values=K_VALUES)

    dropped_total = g_dropped + p_dropped
    rows_total = g_total + p_total
    pct_dropped = 100.0 * dropped_total / rows_total if rows_total else 0.0

    return {
        "convention": convention,
        "n_subjects": len(subject_ids),
        "feature_dim": dim,
        "recall_at_k": {str(k): v for k, v in recall_stats["recall_at_k"].items()},
        "median_rank": float(recall_stats["median_rank"]),
        "mean_rank": float(recall_stats["mean_rank"]),
        "total_queries": recall_stats["total_queries"],
        "pct_rows_dropped": pct_dropped,
        "rows_dropped": dropped_total,
        "rows_total": rows_total,
    }


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(
            f"{_OUT_PATH} already exists; per project rules, write a new filename instead."
        )

    gallery_metadata = json.loads(_GALLERY_METADATA.read_text())

    subject_dirs = sorted(
        (d for d in _GALLERY.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects from {_GALLERY}")

    config = FilterFHEConfig()
    config_weights = {
        "weight_hist_scale0": config.weight_hist_scale0,
        "weight_hist_scale1": config.weight_hist_scale1,
        "weight_row": config.weight_row,
        "weight_spatial": config.weight_spatial,
    }
    print(f"FilterFHEConfig() weights: {config_weights}")

    run_results = {}
    for convention in CONVENTIONS:
        run_results[convention] = run_convention(subject_dirs, config, convention)

    # ---- Side-by-side table ----
    label = {"inverted": "A (inverted mask)", "raw": "B (raw mask)"}
    print("\n" + "=" * 100)
    header = f"{'Metric':<20}" + "".join(f"{label[c]:>25}" for c in CONVENTIONS)
    print(header)
    print("-" * 100)
    for k in K_VALUES:
        row = f"{'Recall@' + str(k):<20}"
        for c in CONVENTIONS:
            row += f"{run_results[c]['recall_at_k'][str(k)]:>24.2f}%"
        print(row)
    for metric_key, metric_label in [
        ("median_rank", "Median rank"),
        ("mean_rank", "Mean rank"),
        ("feature_dim", "Feature dimension"),
        ("pct_rows_dropped", "% rows dropped"),
    ]:
        row = f"{metric_label:<20}"
        for c in CONVENTIONS:
            v = run_results[c][metric_key]
            row += f"{v:>25.2f}" if isinstance(v, float) else f"{v:>25}"
        print(row)
    print("=" * 100)

    out = {
        "metadata": {
            "gallery_path": str(_GALLERY.relative_to(_ROOT)),
            "subject_count": n,
            "gallery_generation_seed": gallery_metadata.get("seed"),
            "gallery_num_processes": gallery_metadata.get("num_processes"),
            "gallery_generator_git_commit": gallery_metadata.get("git_commit"),
            "gallery_metadata_source": str(_GALLERY_METADATA.relative_to(_ROOT)),
            "seeds_used": None,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "config_weights": config_weights,
            "code_shape": "(1, 32, 512) single-scale",
            "min_segment_len": MIN_SEGMENT_LEN,
            "k_values": K_VALUES,
            "source": (
                "christina-fhe-fis/filter_fhe_iris_complete.py "
                "(compute_enhanced_run_stats, compute_l1_distance_plaintext) + "
                "christina-fhe-fis/evaluation.py (compute_recall_at_k), imported read-only"
            ),
            "sampling_note": (
                "Full 2000-subject gallery used, no sub-sampling. This makes the "
                "result reproducible from its own file."
            ),
            "comparable_to": "level1-features/results/christina_rl_maskconvention_300.json "
                              "(same script logic, 300-subject gallery)",
        },
        "conventions": {
            "inverted": "occluded_mask = 1 - sicgen_mask fed to compute_enhanced_run_stats "
                        "(matches non_mask_adjacent_valid's True=occluded expectation)",
            "raw": "sicgen_mask (1=valid) fed as-is to compute_enhanced_run_stats",
        },
        "results": run_results,
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
