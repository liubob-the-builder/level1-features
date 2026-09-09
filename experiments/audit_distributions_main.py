#!/usr/bin/env python3
"""Full genuine/impostor distance DISTRIBUTIONS (as fine-resolution histograms) for the
learned embedding (NN v3, quantized) and Christina's enhanced-RL feature (inverted/fixed
mask convention), on the exact matched protocol used by
all_features_nn_testsplit_comparison.json / audit_nn_v3_dedup_metrics.json /
audit_all_features_dedup_testsplit.json.

Motivation: those files report only summary separation (d', means, recall). This one
preserves the full distribution SHAPE so genuine-vs-impostor separation can be plotted
directly as overlaid histograms.

Protocol (identical to the matched de-duplicated comparison -- nothing re-derived by hand):
  - Probes : the 299 held-out test identities from nn_identity_split.json, stems 2-10,
             MINUS the 5 probes byte-identical to their own gallery image -> 2488 probes.
  - Gallery: all 1998 usable identities (train+val+test), stem '1'.
  - NN      : SSD on the BFV-compatible QUANTIZED integer embedding (quant_range=127,
              embedding_dim=64), via nn_train.ssd_matrix -- the same matrix used for ranking.
  - Christina: L1 via her unmodified compute_l1_distance_plaintext(FilterFHEConfig()),
              inverted mask convention only (occluded_mask = 1 - native_mask).
              NOT SSD -- L1 is what the matched comparison ranked with.

Pair definition (identical to the reference scripts' genuine/impostor split):
  - genuine  = D[probe_i, gallery_column_of_probe_i's_own_identity]  -> 2488 distances
  - impostor = every other cell of D                                 -> 2488 * 1997 distances
  (each identity contributes exactly one gallery entry, stem '1')

Sanity gate: ranks are re-derived from the SAME two distance matrices these histograms
come from, and must reproduce the published de-duplicated numbers exactly, otherwise the
script aborts rather than emitting distributions from a different matrix.

IMPORTANT: the two features use different distance metrics and different value ranges.
Their histograms are NOT on a common axis and are NOT comparable in raw units. Each
feature gets its own bin range; genuine and impostor share bin edges WITHIN a feature
(so they overlay in one panel) but never across features. No normalisation onto a shared
axis is applied.

Read-only w.r.t. every existing checkpoint / split / result file; writes only new
audit_distributions_* filenames. Outputs histograms only -- never the raw distance arrays.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CHRISTINA = _ROOT / "christina-fhe-fis"
_CASIA_CHRISTINA_DIR = _ROOT / "casia-extraction" / "casia-codes-christina"
_CASIA_2D_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS = _LEVEL1 / "results"
_SPLIT_PATH = _RESULTS / "nn_identity_split.json"
_NN_DEDUP_REF = _RESULTS / "audit_nn_v3_dedup_metrics.json"
_ALLFEAT_DEDUP_REF = _RESULTS / "audit_all_features_dedup_testsplit.json"

_OUT_JSON = _RESULTS / "audit_distributions_results.json"
_OUT_MD = _RESULTS / "audit_distributions_RESULTS.md"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_CHRISTINA))

# Reuse the already-audited loaders/embedders/metrics verbatim rather than reimplementing.
import audit_maskleak_main as m1                       # noqa: E402  (load_common, embed_baseline)
import audit_all_features_dedup_testsplit as m2        # noqa: E402  (Christina helpers)
from stage2_evaluate import eer, d_prime               # noqa: E402  (imported unmodified)
from nn_split import load_split                        # noqa: E402

from filter_fhe_iris_complete import (                 # noqa: E402
    FilterFHEConfig,
    compute_l1_distance_plaintext,
    compute_template_features,
    extract_feature_vector_all_scales,
)

K_VALUES = [10, 50, 100, 300]
EXPECTED_N_GALLERY = 1998
EXPECTED_N_PROBES = 2488
N_BINS_SHARED = 512   # genuine + impostor share these edges WITHIN a feature (overlay panel)
N_BINS_OWN = 256      # each set additionally binned over its OWN min/max (shape detail)

NN_KEY = "NN (v3, learned, quantized)"
CHRISTINA_KEY = "Christina RL (inverted)"

_RANK_FIELDS = ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300",
                "median_rank", "mean_rank")


# ---------------------------------------------------------------------------
# Genuine / impostor extraction + histogramming
# ---------------------------------------------------------------------------

def split_genuine_impostor(D: np.ndarray, gallery_ids: list, probe_true_ids: list):
    """genuine = each probe's distance to its OWN identity's single gallery entry;
    impostor = that probe's distance to every OTHER identity's gallery entry.
    Same construction as the reference comparison scripts."""
    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(D.shape[0])

    genuine = D[row_idx, true_col].astype(np.float64)
    impostor_mask = np.ones(D.shape, dtype=bool)
    impostor_mask[row_idx, true_col] = False
    impostor = D[impostor_mask].astype(np.float64)

    assert genuine.size == D.shape[0]
    assert impostor.size == D.shape[0] * (D.shape[1] - 1)
    return genuine, impostor


def set_summary(x: np.ndarray) -> dict:
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),          # population std (ddof=0), matching d_prime's .var()
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def histogram(x: np.ndarray, lo: float, hi: float, n_bins: int) -> dict:
    counts, edges = np.histogram(x, bins=n_bins, range=(float(lo), float(hi)))
    assert int(counts.sum()) == int(x.size), "histogram dropped samples outside its range"
    return {
        "n_bins": int(n_bins),
        "range": [float(edges[0]), float(edges[-1])],
        "bin_edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def feature_block(name: str, metric: str, metric_detail: str, dim: int,
                  genuine: np.ndarray, impostor: np.ndarray, rank_metrics: dict) -> dict:
    lo = float(min(genuine.min(), impostor.min()))
    hi = float(max(genuine.max(), impostor.max()))

    shared_gen = histogram(genuine, lo, hi, N_BINS_SHARED)
    shared_imp = histogram(impostor, lo, hi, N_BINS_SHARED)
    assert shared_gen["bin_edges"] == shared_imp["bin_edges"], "shared-edge histograms diverged"

    block = {
        "distance_metric": metric,
        "distance_metric_detail": metric_detail,
        "feature_dim": dim,
        "n_gallery": rank_metrics["n_gallery"],
        "n_probes": rank_metrics["n_probes"],
        "axis_note": (f"Bin edges below are specific to {name} and its own distance metric "
                      f"({metric}). Genuine and impostor share edges WITHIN this feature so "
                      "they overlay in one panel; edges are NOT shared with the other feature "
                      "and the two features' raw distance values are not comparable."),
        "summary": {
            "genuine": set_summary(genuine),
            "impostor": set_summary(impostor),
            "d_prime": d_prime(genuine, impostor),
            "eer": eer(genuine, impostor),
        },
        "rank_metrics_from_same_matrix": rank_metrics,
        "histograms": {
            "shared_edges": {
                "description": "Genuine and impostor binned on IDENTICAL edges spanning "
                               "min(genuine union impostor) .. max(genuine union impostor). "
                               "Use this pair for the overlaid separation plot.",
                "n_bins": N_BINS_SHARED,
                "range": [lo, hi],
                "bin_edges": shared_gen["bin_edges"],
                "genuine_counts": shared_gen["counts"],
                "impostor_counts": shared_imp["counts"],
            },
            "own_range": {
                "description": "Each set additionally binned over its OWN min..max, so the "
                               "genuine distribution's shape survives even where the impostor "
                               "tail dominates the shared range. Edges differ between the two "
                               "sets here -- do NOT overlay these two directly.",
                "genuine": histogram(genuine, genuine.min(), genuine.max(), N_BINS_OWN),
                "impostor": histogram(impostor, impostor.min(), impostor.max(), N_BINS_OWN),
            },
        },
    }
    return block


def check_against_reference(label: str, got: dict, ref: dict):
    mismatches = {k: (got[k], ref[k]) for k in _RANK_FIELDS
                  if not np.isclose(got[k], ref[k], atol=1e-9, rtol=0.0)}
    if mismatches:
        raise RuntimeError(
            f"SANITY GATE FAILED for {label}: ranks re-derived from this distance matrix do "
            f"not reproduce the published de-duplicated numbers: {mismatches}. Refusing to "
            "emit distributions from a matrix that differs from the ranked one."
        )
    print(f"[sanity] {label}: re-derived ranks match the published de-duplicated numbers exactly.",
          flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for p in (_OUT_JSON, _OUT_MD):
        if p.exists():
            raise FileExistsError(f"{p} already exists; write a new filename instead.")

    split = load_split(_SPLIT_PATH)
    nn_ref = json.loads(_NN_DEDUP_REF.read_text())["quantized_deduplicated"]
    christina_ref = json.loads(_ALLFEAT_DEDUP_REF.read_text())["results"]["Christina (inverted)"]

    features = {}

    # ================= 1. Learned embedding (NN v3, quantized) -- SSD =================
    print("=== NN v3: loading split/checkpoint/templates via audit_maskleak_main.load_common() ===",
          flush=True)
    test_ids, full_gallery_ids, gallery_items, probe_items, model = m1.load_common()
    assert len(full_gallery_ids) == EXPECTED_N_GALLERY
    assert len(probe_items) == EXPECTED_N_PROBES
    print(f"gallery={len(gallery_items)} probes={len(probe_items)}", flush=True)

    t0 = time.time()
    gallery_emb_f, nn_gallery_ids = m1.embed_baseline(model, gallery_items, is_probe=False)
    probe_emb_f, nn_probe_true_ids = m1.embed_baseline(model, probe_items, is_probe=True)
    gallery_emb_q = model.quantize(torch.from_numpy(gallery_emb_f)).numpy().astype(np.float64)
    probe_emb_q = model.quantize(torch.from_numpy(probe_emb_f)).numpy().astype(np.float64)
    print(f"embedded + quantized in {time.time()-t0:.1f}s", flush=True)

    D_nn = m1.ssd_matrix(gallery_emb_q, probe_emb_q)
    assert D_nn.shape == (EXPECTED_N_PROBES, EXPECTED_N_GALLERY), D_nn.shape

    ranks_nn = m1.ranks_of_true_identity(D_nn, nn_gallery_ids, nn_probe_true_ids)
    nn_rank_metrics = {
        "n_gallery": len(nn_gallery_ids), "n_probes": len(nn_probe_true_ids),
        "median_rank": float(np.median(ranks_nn)), "mean_rank": float(np.mean(ranks_nn)),
        **{f"recall_at_{k}": m1.recall_at_k(ranks_nn, k) for k in K_VALUES},
    }
    check_against_reference(NN_KEY, nn_rank_metrics, nn_ref)

    gen_nn, imp_nn = split_genuine_impostor(D_nn, nn_gallery_ids, nn_probe_true_ids)
    print(f"NN genuine={gen_nn.size} impostor={imp_nn.size}", flush=True)
    features[NN_KEY] = feature_block(
        NN_KEY, "SSD",
        "Sum of squared differences (nn_train.ssd_matrix) on the BFV-compatible QUANTIZED "
        "integer embedding produced by CircularConvEncoder.quantize "
        "(quant_range=127, embedding_dim=64), checkpoint nn_level1_v3_checkpoint_best.pt.",
        int(gallery_emb_q.shape[1]), gen_nn, imp_nn, nn_rank_metrics)

    del D_nn, gallery_items, probe_items, gallery_emb_f, probe_emb_f, gallery_emb_q, probe_emb_q

    # ================= 2. Christina's RL feature (inverted) -- L1 =================
    print("\n=== Christina RL (inverted convention): extracting features ===", flush=True)
    config = FilterFHEConfig()
    t0 = time.time()

    gallery_feats = {}
    for identity in full_gallery_ids:
        d = _CASIA_CHRISTINA_DIR / identity
        gcode, gmask_native = m2.load_christina_pair(d, "1")
        gallery_feats[identity] = compute_template_features(
            gcode, m2.mask_for_convention(gmask_native, "inverted"))
    c_gallery_ids = list(gallery_feats.keys())

    c_probe_true_ids, c_probe_feats = [], []
    skipped = 0
    for identity in test_ids:
        d = _CASIA_CHRISTINA_DIR / identity
        for stem in m2.available_probe_stems_christina(d):
            if (identity, stem) in m2.DUPLICATE_PROBES:
                skipped += 1
                continue
            pcode, pmask_native = m2.load_christina_pair(d, stem)
            c_probe_true_ids.append(identity)
            c_probe_feats.append(compute_template_features(
                pcode, m2.mask_for_convention(pmask_native, "inverted")))
    assert skipped == 5, f"expected to skip 5 duplicate probes, skipped {skipped}"
    assert len(c_probe_feats) == EXPECTED_N_PROBES, len(c_probe_feats)
    assert len(c_gallery_ids) == EXPECTED_N_GALLERY

    christina_dim = len(extract_feature_vector_all_scales(next(iter(gallery_feats.values()))))
    print(f"  {len(c_gallery_ids)} galleries, {len(c_probe_feats)} probes (skipped {skipped}), "
          f"dim={christina_dim} (extraction {time.time()-t0:.1f}s)", flush=True)

    print("  computing the full 2488 x 1998 L1 distance matrix "
          "(her unmodified compute_l1_distance_plaintext)...", flush=True)
    t0 = time.time()
    n_p, n_g = len(c_probe_feats), len(c_gallery_ids)
    D_c = np.empty((n_p, n_g), dtype=np.float64)
    gallery_feat_list = [gallery_feats[gid] for gid in c_gallery_ids]
    for i, pfeat in enumerate(c_probe_feats):
        row = D_c[i]
        for j, gfeat in enumerate(gallery_feat_list):
            row[j] = compute_l1_distance_plaintext(pfeat, gfeat, config)
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{n_p} probes ({el:.1f}s elapsed, "
                  f"~{el/(i+1)*(n_p-i-1):.0f}s remaining)", flush=True)
    print(f"  distance matrix done in {time.time()-t0:.1f}s", flush=True)

    # Rank exactly as the reference script did: stable sort of (gallery_id, distance) pairs
    # built in gallery order, then the 1-indexed position of the true identity.
    ranks_c = np.empty(n_p, dtype=np.int64)
    for i in range(n_p):
        scores = [(gid, D_c[i, j]) for j, gid in enumerate(c_gallery_ids)]
        scores.sort(key=lambda x: x[1])
        ranks_c[i] = [gid for gid, _ in scores].index(c_probe_true_ids[i]) + 1
    c_rank_metrics = {
        "n_gallery": n_g, "n_probes": n_p,
        "median_rank": float(np.median(ranks_c)), "mean_rank": float(np.mean(ranks_c)),
        **{f"recall_at_{k}": m1.recall_at_k(ranks_c, k) for k in K_VALUES},
    }
    check_against_reference(CHRISTINA_KEY, c_rank_metrics, christina_ref)

    gen_c, imp_c = split_genuine_impostor(D_c, c_gallery_ids, c_probe_true_ids)
    print(f"Christina genuine={gen_c.size} impostor={imp_c.size}", flush=True)
    features[CHRISTINA_KEY] = feature_block(
        CHRISTINA_KEY, "L1 (weighted)",
        "Weighted L1 via Christina's unmodified compute_l1_distance_plaintext with "
        "FilterFHEConfig() defaults: 0.50*hist(scale_0) + 0.20*sum(hist(scale_1..31)) + "
        "0.20*mean(row_diffs) + 0.10*mean(spatial_diffs), aggregated over 32 scales. "
        "This is the metric the matched comparison ranked with -- NOT SSD. Mask convention: "
        "inverted/fixed (occluded_mask = 1 - native_mask).",
        christina_dim, gen_c, imp_c, c_rank_metrics)

    del D_c

    # ================= Output =================
    out = {
        "metadata": {
            "purpose": "Full genuine and impostor distance DISTRIBUTIONS (fine-resolution "
                       "histograms) for the learned embedding and Christina's RL feature, so "
                       "separation can be plotted directly rather than only summarised by d' "
                       "and means.",
            "protocol": "Identical to the matched de-duplicated comparison "
                        "(all_features_nn_testsplit_comparison.json / "
                        "audit_nn_v3_dedup_metrics.json / audit_all_features_dedup_testsplit.json): "
                        "299 held-out test identities as probes, ranked against the full "
                        "1998-identity gallery, de-duplicated to 2488 probes.",
            "gallery_paths": {
                NN_KEY: str(_CASIA_2D_DIR.relative_to(_ROOT)),
                CHRISTINA_KEY: str(_CASIA_CHRISTINA_DIR.relative_to(_ROOT)),
            },
            "n_gallery_identities": EXPECTED_N_GALLERY,
            "n_test_identities_probes": len(test_ids),
            "n_probes": EXPECTED_N_PROBES,
            "subject_counts": {
                "gallery_subjects": EXPECTED_N_GALLERY,
                "probe_subjects": len(test_ids),
                "probes_after_deduplication": EXPECTED_N_PROBES,
            },
            "duplicate_probes_excluded": sorted(f"{a}_{b}" for a, b in m2.DUPLICATE_PROBES),
            "duplicate_exclusion_reason": "byte-identical to their own gallery image "
                                          "(CASIA-Iris-Thousand source-data artifact, see "
                                          "audit_nn_v3_verification.json check_2).",
            "excluded_identities": ["524_L", "668_L"],
            "pair_definition": {
                "genuine": "distance from each probe to its OWN identity's single gallery "
                           "entry (stem '1') -> one per probe, 2488 total.",
                "impostor": "distance from each probe to every OTHER identity's gallery entry "
                            "-> 2488 * 1997 = 4968536 total.",
            },
            "distance_metrics": {
                NN_KEY: "SSD on the quantized integer embedding",
                CHRISTINA_KEY: "L1 (compute_l1_distance_plaintext), inverted mask convention",
            },
            "NOT_COMPARABLE_WARNING": (
                "The two features use DIFFERENT distance metrics (SSD vs weighted L1) and "
                "DIFFERENT value ranges. Their histograms are NOT on a common axis and their "
                "raw distance values are NOT directly comparable. Each feature's histograms use "
                "its own bin range; genuine and impostor share bin edges only WITHIN a feature. "
                "No normalisation onto a shared axis has been applied. Compare the features via "
                "d'/EER/recall, never by overlaying their raw distance histograms."
            ),
            "histogram_settings": {
                "shared_edges_n_bins": N_BINS_SHARED,
                "own_range_n_bins": N_BINS_OWN,
                "raw_distances_emitted": False,
                "note": "Histogram data only -- raw distance arrays are deliberately not written.",
            },
            "sanity_gate": {
                "description": "Ranks were re-derived from the SAME distance matrices these "
                               "histograms come from and had to reproduce the published "
                               "de-duplicated numbers exactly (atol=1e-9), else the run aborts.",
                "nn_reference": "level1-features/results/audit_nn_v3_dedup_metrics.json "
                                "(quantized_deduplicated)",
                "christina_reference": "level1-features/results/audit_all_features_dedup_testsplit.json "
                                       "(results['Christina (inverted)'])",
                "passed": True,
            },
            "split_path": str(_SPLIT_PATH.relative_to(_ROOT)),
            "seeds": {
                "identity_split_seed": split["metadata"]["seed"],
                "experiment_rng": "none -- this experiment is fully deterministic, no sampling "
                                  "or subsampling is performed.",
            },
            "nn_checkpoint": "level1-features/results/nn_level1_v3_checkpoint_best.pt",
            "christina_config_weights": {
                "weight_hist_scale0": config.weight_hist_scale0,
                "weight_hist_scale1": config.weight_hist_scale1,
                "weight_row": config.weight_row,
                "weight_spatial": config.weight_spatial,
            },
            "git_commit": m1.get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "features": features,
    }

    with open(_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_JSON}", flush=True)

    _OUT_MD.write_text(build_markdown(out))
    print(f"Saved {_OUT_MD}", flush=True)

    print("\n" + "=" * 100)
    for name, blk in features.items():
        s = blk["summary"]
        print(f"{name}: metric={blk['distance_metric']}  d'={s['d_prime']:.4f}  "
              f"EER={s['eer']:.4f}\n   genuine  mean={s['genuine']['mean']:.4f} "
              f"std={s['genuine']['std']:.4f} range=[{s['genuine']['min']:.4f}, "
              f"{s['genuine']['max']:.4f}]\n   impostor mean={s['impostor']['mean']:.4f} "
              f"std={s['impostor']['std']:.4f} range=[{s['impostor']['min']:.4f}, "
              f"{s['impostor']['max']:.4f}]")
    print("=" * 100)


def build_markdown(out: dict) -> str:
    md = out["metadata"]
    L = []
    L.append("# Genuine / impostor distance distributions -- learned embedding vs Christina's RL")
    L.append("")
    L.append(f"Date: {md['date']}  |  git commit: `{md['git_commit']}`")
    L.append("")
    L.append("Full genuine and impostor distance distributions, emitted as fine-resolution "
             "histograms so separation can be plotted directly instead of being summarised only "
             "by d' and means. Histogram data lives in "
             "`level1-features/results/audit_distributions_results.json`.")
    L.append("")
    L.append("## Protocol")
    L.append("")
    L.append(f"- Probes: {md['n_test_identities_probes']} held-out test identities, stems 2-10, "
             f"de-duplicated to **{md['n_probes']} probes** "
             f"(excluded: {', '.join(md['duplicate_probes_excluded'])} -- "
             f"{md['duplicate_exclusion_reason']}).")
    L.append(f"- Gallery: **{md['n_gallery_identities']} identities**, stem `1` "
             f"(train+val+test; `524_L`/`668_L` already absent from the split).")
    L.append(f"- Genuine pairs: {md['pair_definition']['genuine']}")
    L.append(f"- Impostor pairs: {md['pair_definition']['impostor']}")
    L.append("- Sanity gate: ranks re-derived from these exact distance matrices reproduce the "
             "published de-duplicated numbers to `atol=1e-9`.")
    L.append("")
    L.append("## :warning: The two features are NOT on a common axis")
    L.append("")
    L.append(md["NOT_COMPARABLE_WARNING"])
    L.append("")
    L.append("## Metric and value range per feature")
    L.append("")
    L.append("| Feature | Metric | Dim | Genuine range | Impostor range | Shared bin range (plot axis) |")
    L.append("|---|---|---|---|---|---|")
    for name, blk in out["features"].items():
        g, i = blk["summary"]["genuine"], blk["summary"]["impostor"]
        r = blk["histograms"]["shared_edges"]["range"]
        L.append(f"| {name} | {blk['distance_metric']} | {blk['feature_dim']} | "
                 f"[{g['min']:.4g}, {g['max']:.4g}] | [{i['min']:.4g}, {i['max']:.4g}] | "
                 f"[{r[0]:.4g}, {r[1]:.4g}] |")
    L.append("")
    L.append("## Separation summary")
    L.append("")
    L.append("| Feature | Genuine mean +/- std | Impostor mean +/- std | d' | EER | median rank | R@10 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, blk in out["features"].items():
        s, rm = blk["summary"], blk["rank_metrics_from_same_matrix"]
        L.append(f"| {name} | {s['genuine']['mean']:.4g} +/- {s['genuine']['std']:.4g} | "
                 f"{s['impostor']['mean']:.4g} +/- {s['impostor']['std']:.4g} | "
                 f"{s['d_prime']:.4f} | {s['eer']:.4f} | {rm['median_rank']:.1f} | "
                 f"{rm['recall_at_10']:.4f} |")
    L.append("")
    L.append("d' and EER are computed on the same genuine/impostor sets the histograms describe, "
             "using the unmodified `stage2_evaluate.d_prime` / `stage2_evaluate.eer`. Because the "
             "metrics differ, d' and EER (both scale-free) are the valid cross-feature "
             "comparison; the raw distance means are not.")
    L.append("")
    L.append("## Metric detail")
    L.append("")
    for name, blk in out["features"].items():
        L.append(f"- **{name}** -- {blk['distance_metric_detail']}")
    L.append("")
    L.append("## How to plot")
    L.append("")
    L.append("For each feature, one panel: `histograms.shared_edges.bin_edges` "
             f"({N_BINS_SHARED} bins) with `genuine_counts` and `impostor_counts` overlaid. "
             "`histograms.own_range` gives each set re-binned over its own min..max "
             f"({N_BINS_OWN} bins) for shape detail -- those two must not be overlaid on each "
             "other. Never place the two features in the same panel or on a shared axis.")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
