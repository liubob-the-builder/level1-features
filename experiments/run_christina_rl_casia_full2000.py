#!/usr/bin/env python3
"""Full-scale (1000-subject / 2000-identity) real-CASIA replication of Christina's
enhanced-RL Level-1 feature, extending run_christina_rl_casia.py's 300-identity run
to the complete CASIA-Iris-Thousand extraction, to check whether her feature
reproduces the paper's reported real-CASIA RL Recall@50 = 38.3% at matched gallery
scale (the paper's own reported figure is presumably closer to a 1000-subject scale
than the earlier 300-identity subsample).

Data: casia-extraction/casia-codes-christina/<identity>/{stem}_code.npy,
{stem}_mask.npy -- raw (32, 512) int32 arrays exactly as extract_iris_code() returns
them (mask 1=valid, Open Iris's native convention). Fed to compute_template_features
with NO reshaping, reproducing her pipeline's real n_scales=32 pseudo-scale behavior.
All 2000 identities from casia-extraction/manifest.json, EXCEPT identities 524_L and
668_L, which failed to produce a usable gallery image (image 00) during extraction --
excluded here exactly as in the other full2000 evaluations
(eval_myfeatures_realcasia_full2000.py, eval_fusion_realcasia_full2000.py), leaving
1998 usable galleries and 16,457 probes. This is directly comparable to those runs.

Two runs, varying only the mask convention fed into compute_enhanced_run_stats (via
compute_template_features):
  Run RAW      -- mask fed as-is (1=valid), exactly as her real pipeline does (no
                  inversion anywhere in extract_iris_code or enroll/identify).
  Run INVERTED -- occluded_mask = 1 - mask fed instead, matching what
                  non_mask_adjacent_valid's own logic expects (True=occluded).

For each run: every probe is ranked against all 1998 galleries by
compute_l1_distance_plaintext (her function, default FilterFHEConfig()), and the
per-probe rankings are fed into compute_recall_at_k (christina-fhe-fis/evaluation.py,
also imported read-only) for Recall@10/20/50/100/150/300, median/mean rank. Also
reports feature dimension and % of pseudo-scales that produced no valid segment
>= min_segment_len=8 (dropped rows), and compares Recall@50 against the paper's
reported CASIA RL figure (38.3%).
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

_LEVEL1 = Path(__file__).resolve().parent.parent
_ROOT = _LEVEL1.parent
_CHRISTINA = _ROOT / "christina-fhe-fis"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_CHRISTINA_DIR = _CASIA_DIR / "casia-codes-christina"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_PATH = _RESULTS / "christina_realcasia_full2000.json"

sys.path.insert(0, str(_CHRISTINA))

import numpy as np  # noqa: E402

from filter_fhe_iris_complete import (  # noqa: E402
    FilterFHEConfig,
    compute_l1_distance_plaintext,
    compute_template_features,
    extract_feature_vector_all_scales,
)
from evaluation import compute_recall_at_k  # noqa: E402

MIN_SEGMENT_LEN = 8  # matches compute_enhanced_run_stats's default
K_VALUES = [10, 20, 50, 100, 150, 300]
CONVENTIONS = ("raw", "inverted")
PROBE_STEMS = [str(i) for i in range(2, 11)]  # stems "2".."10" = probe images 01-09
PAPER_CASIA_RECALL_AT_50 = 38.3  # Karakosta & Knottenbelt's reported RL Recall@50 on real CASIA
EXCLUDED_IDENTITIES = ["524_L", "668_L"]  # failed gallery image 00 during extraction
EXPECTED_N_GALLERY = 1998
EXPECTED_N_PROBES = 16457


def load_manifest():
    return json.loads(_CASIA_MANIFEST.read_text())


def available_probe_stems(identity_dir: Path) -> list:
    return [
        s for s in PROBE_STEMS
        if (identity_dir / f"{s}_code.npy").exists() and (identity_dir / f"{s}_mask.npy").exists()
    ]


def load_pair(identity_dir: Path, stem: str):
    code = np.load(identity_dir / f"{stem}_code.npy")
    mask = np.load(identity_dir / f"{stem}_mask.npy")
    return code, mask


def mask_for_convention(native_mask: np.ndarray, convention: str) -> np.ndarray:
    """native_mask: Open Iris's native convention, 1=valid, 0=occluded."""
    if convention == "inverted":
        return 1 - native_mask  # 1=occluded, matches non_mask_adjacent_valid's expectation
    elif convention == "raw":
        return native_mask  # fed as-is, 1=valid -- what her real pipeline actually does
    raise ValueError(convention)


def count_dropped_rows(features: dict) -> tuple:
    """Pseudo-scales (rows) with mean_run_0==0 and mean_run_1==0: no segment
    >= MIN_SEGMENT_LEN contributed (includes pseudo-scales with zero valid bits)."""
    dropped = 0
    total = 0
    for stats in features.values():
        for row in stats["row_stats"]:
            total += 1
            if row["mean_run_0"] == 0 and row["mean_run_1"] == 0:
                dropped += 1
    return dropped, total


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


def run_convention(identities: list, config: FilterFHEConfig, convention: str) -> dict:
    print(f"\n=== Convention: {convention} ===")

    gallery_feats = {}
    probe_records = []  # list of (true_identity, features)
    dropped_total = 0
    rows_total = 0

    for ident in identities:
        identity = ident["identity"]
        if identity in EXCLUDED_IDENTITIES:
            continue
        if not ident["gallery_ok"]:
            continue  # defensive; should already match EXCLUDED_IDENTITIES
        identity_dir = _CASIA_CHRISTINA_DIR / identity

        gcode, gmask_native = load_pair(identity_dir, "1")
        gmask = mask_for_convention(gmask_native, convention)
        gfeat = compute_template_features(gcode, gmask)
        gallery_feats[identity] = gfeat
        d, t = count_dropped_rows(gfeat)
        dropped_total += d
        rows_total += t

        for stem in available_probe_stems(identity_dir):
            pcode, pmask_native = load_pair(identity_dir, stem)
            pmask = mask_for_convention(pmask_native, convention)
            pfeat = compute_template_features(pcode, pmask)
            probe_records.append((identity, pfeat))
            d, t = count_dropped_rows(pfeat)
            dropped_total += d
            rows_total += t

    gallery_ids = list(gallery_feats.keys())
    dim = len(extract_feature_vector_all_scales(next(iter(gallery_feats.values()))))
    print(f"  {len(gallery_ids)} galleries, {len(probe_records)} probes, feature dim={dim}")

    results = []
    for i, (true_identity, pfeat) in enumerate(probe_records):
        level1_scores = [
            (gid, compute_l1_distance_plaintext(pfeat, gallery_feats[gid], config))
            for gid in gallery_ids
        ]
        level1_scores.sort(key=lambda x: x[1])
        results.append({"true_identity": true_identity, "level1_scores": level1_scores})
        if (i + 1) % 1000 == 0:
            print(f"    ranked {i + 1}/{len(probe_records)} probes...")

    recall_stats = compute_recall_at_k(results, k_values=K_VALUES)

    pct_dropped = 100.0 * dropped_total / rows_total if rows_total else 0.0

    return {
        "convention": convention,
        "n_galleries": len(gallery_ids),
        "n_probes": len(probe_records),
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

    manifest = load_manifest()
    identities = manifest["identities"]
    n_identities_total = len(identities)
    usable = [i for i in identities if i["identity"] not in EXCLUDED_IDENTITIES and i["gallery_ok"]]
    n_probes_expected = sum(i["n_probes_ok"] for i in usable)
    print(f"Loaded manifest: {n_identities_total} identities total, "
          f"{len(usable)} usable (excluding {EXCLUDED_IDENTITIES}), "
          f"{n_probes_expected} expected probes")

    assert len(usable) == EXPECTED_N_GALLERY, (
        f"Expected {EXPECTED_N_GALLERY} usable galleries, found {len(usable)}"
    )
    assert n_probes_expected == EXPECTED_N_PROBES, (
        f"Expected {EXPECTED_N_PROBES} probes, found {n_probes_expected}"
    )

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
        run_results[convention] = run_convention(identities, config, convention)

    # ---- side-by-side table ----
    label = {"raw": "RAW (1=valid, faithful)", "inverted": "INVERTED (fixed convention)"}
    print("\n" + "=" * 100)
    header = f"{'Metric':<20}" + "".join(f"{label[c]:>28}" for c in CONVENTIONS)
    print(header)
    print("-" * 100)
    for k in K_VALUES:
        row = f"{'Recall@' + str(k):<20}"
        for c in CONVENTIONS:
            row += f"{run_results[c]['recall_at_k'][str(k)]:>27.2f}%"
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
            row += f"{v:>28.2f}" if isinstance(v, float) else f"{v:>28}"
        print(row)
    print("=" * 100)

    paper_comparison = {
        "paper_recall_at_50_pct": PAPER_CASIA_RECALL_AT_50,
        "paper_source": "Karakosta & Knottenbelt, 'Iris Through the Looking Glass', RL feature, real CASIA",
    }
    for convention in CONVENTIONS:
        r50 = run_results[convention]["recall_at_k"]["50"]
        paper_comparison[convention] = {
            "recall_at_50_pct": r50,
            "delta_vs_paper_pct_points": r50 - PAPER_CASIA_RECALL_AT_50,
        }
    print("\nPaper comparison (RL Recall@50 on real CASIA):")
    print(f"  Paper:    {PAPER_CASIA_RECALL_AT_50:.1f}%")
    for convention in CONVENTIONS:
        pc = paper_comparison[convention]
        print(f"  {label[convention]:<28}: {pc['recall_at_50_pct']:.2f}% (delta {pc['delta_vs_paper_pct_points']:+.2f} pts)")

    out = {
        "metadata": {
            "data_source": "casia-extraction/casia-codes-christina/ (real CASIA-Iris-Thousand via Open Iris, full 1000-subject/2000-identity extraction)",
            "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
            "identity_count_total": n_identities_total,
            "identity_count_usable": len(usable),
            "excluded_identities": EXCLUDED_IDENTITIES,
            "excluded_reason": "failed to produce a usable gallery image (image 00) during extraction",
            "subject_selection_rule": manifest.get("subject_selection_rule"),
            "n_probes_expected_from_manifest": n_probes_expected,
            "seeds_used": None,
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
            "config_weights": config_weights,
            "code_shape": "(32, 512) int32, fed to compute_template_features with NO reshaping (n_scales=32 pseudo-scales)",
            "min_segment_len": MIN_SEGMENT_LEN,
            "k_values": K_VALUES,
            "source": (
                "christina-fhe-fis/filter_fhe_iris_complete.py "
                "(compute_template_features, compute_l1_distance_plaintext) + "
                "christina-fhe-fis/evaluation.py (compute_recall_at_k), imported read-only"
            ),
            "comparable_to": [
                "level1-features/results/myfeatures_realcasia_full2000.json",
                "level1-features/results/fusion_realcasia_full2000.json",
            ],
        },
        "conventions": {
            "raw": "mask fed as-is (1=valid) to compute_enhanced_run_stats -- the faithful, "
                   "unmodified reproduction of her real pipeline (extract_iris_code performs no inversion)",
            "inverted": "occluded_mask = 1 - native_mask fed instead, matching "
                        "non_mask_adjacent_valid's True=occluded expectation",
        },
        "results": run_results,
        "paper_comparison": paper_comparison,
    }

    _RESULTS.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
