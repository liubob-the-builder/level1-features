#!/usr/bin/env python3
"""Fully-held-out gallery audit: does the learned NN embedding's advantage over the
hand-designed features (DFT/AC/RL-C9/RL-C6) and Christina's RL feature survive when the
gallery is restricted to ONLY the 299 held-out test identities (no train/val identities
as distractors), so that neither the probes NOR the gallery distractors were seen during
training?

Motivation: the NN was trained on the 1699 train+val identities, which appear as
distractors in the standard 1998-identity gallery. Restricting the gallery to the 299
test identities removes that possible confound. The 299-identity gallery is a strictly
easier ranking task for every feature (fewer distractors), so absolute recall rises for
all of them; the quantity of interest is whether the NN-vs-hand-designed-feature GAP is
preserved, not the absolute numbers themselves.

Read-only w.r.t. every existing file, checkpoint, and result. Reuses (imports, never
edits):
  - audit_maskleak_main.py (as m1): load_common() (299-test-identity / 1998-gallery /
    2488-de-duplicated-probe protocol), embed_baseline(), rank_metrics(), and (via m1's
    own module-level imports) nn_train.ssd_matrix / ranks_of_true_identity /
    recall_at_k, nn_model.CircularConvEncoder, get_git_commit().
  - audit_embcompare_main.py (as m2): extract_features_maskaware() (DFT bins 30-50,
    AC concat, RL-C9, RL-C6 -- mask-aware, same convention throughout this project).
  - audit_all_features_dedup_testsplit.py (as m3): load_mask_2d(), load_template
    (via dft_spectrum), load_christina_pair(), mask_for_convention(),
    available_probe_stems_2d(), available_probe_stems_christina(), DUPLICATE_PROBES,
    _CASIA_2D_DIR, _CASIA_CHRISTINA_DIR.
  - christina-fhe-fis/filter_fhe_iris_complete.py: FilterFHEConfig,
    compute_template_features, compute_l1_distance_plaintext (unmodified).

The ONLY change from the matched comparison is the gallery: for every feature, gallery
vectors are restricted to the 299 test identities (instead of the full 1998), while the
probe set (2488, de-duplicated), feature extraction, and distance metric (SSD for
NN/DFT/AC/RL-C9/RL-C6, L1 for Christina) are held byte-for-byte identical.

Every feature's full-1998-gallery numbers are recomputed here (same functions, same
protocol) and checked against the stored reference file before its 299-gallery number is
trusted; on any mismatch the script raises rather than reporting a possibly-wrong number.
Also verifies, per feature, that every probe's genuine-match identity is actually present
in the restricted 299-identity gallery (it must be, since every probe's true identity IS
a test identity -- this is a structural check, not expected to fail).
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
_RESULTS = _LEVEL1 / "results"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

import audit_maskleak_main as m1  # noqa: E402 -- reused unmodified
import audit_embcompare_main as m2  # noqa: E402 -- reused unmodified
import audit_all_features_dedup_testsplit as m3  # noqa: E402 -- reused unmodified

from filter_fhe_iris_complete import (  # noqa: E402 -- unmodified, christina-fhe-fis
    FilterFHEConfig,
    compute_l1_distance_plaintext,
    compute_template_features,
)

_OUT_JSON = _RESULTS / "audit_heldout_gallery_results.json"
_OUT_MD = _RESULTS / "audit_heldout_gallery_RESULTS.md"

K_VALUES = [10, 50, 100, 300]
HAND_FEATURE_NAMES = m2.HAND_FEATURE_NAMES  # ("DFT", "AC", "RL-C9", "RL-C6")
EXPECTED_N_TEST_IDENTITIES = 299
EXPECTED_N_GALLERY_FULL = 1998
EXPECTED_N_GALLERY_HELDOUT = 299
EXPECTED_N_PROBES = 2488
_HAND_FEATURE_REF_PATH = _RESULTS / "audit_all_features_dedup_testsplit.json"
_ATOL = 1e-9


def dense_ranks(D: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(D, axis=1), axis=1)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def metrics_from_ranks(ranks: np.ndarray, n_gallery: int, n_probes: int) -> dict:
    out = {"n_gallery": n_gallery, "n_probes": n_probes,
           "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks))}
    for k in K_VALUES:
        out[f"recall_at_{k}"] = recall_at_k(ranks, k)
    return out


def check_reference_match(name: str, computed: dict, reference: dict, keys) -> dict:
    mismatches = {k: (computed[k], reference[k]) for k in keys
                  if not np.isclose(computed[k], reference[k], atol=_ATOL)}
    if mismatches:
        raise RuntimeError(f"{name}: recomputed full-1998-gallery metrics do not match "
                            f"stored reference: {mismatches}")
    return {"matches_reference": True}


def verify_probe_coverage(feature_name: str, gallery_ids, probe_true_ids) -> dict:
    missing = sorted(set(probe_true_ids) - set(gallery_ids))
    result = {"n_probes": len(probe_true_ids), "n_missing_genuine_match": len(missing),
              "missing_probe_identities": missing, "all_present": len(missing) == 0}
    if missing:
        print(f"  !!! {feature_name}: {len(missing)} probes have NO genuine match in the "
              f"heldout gallery: {missing}")
    else:
        print(f"  [verified] {feature_name}: every probe's genuine-match identity is present "
              f"in the {len(set(gallery_ids))}-identity heldout gallery.")
    return result


def main():
    if _OUT_JSON.exists():
        raise FileExistsError(f"{_OUT_JSON} already exists; write a new filename instead.")

    hand_ref = json.loads(_HAND_FEATURE_REF_PATH.read_text())["results"]

    print("=== loading split / checkpoint / raw templates (via audit_maskleak_main.load_common) ===")
    test_ids, full_gallery_ids, gallery_items, probe_items, model = m1.load_common()
    heldout_gallery_ids_target = sorted(test_ids)
    assert len(test_ids) == EXPECTED_N_TEST_IDENTITIES
    assert len(full_gallery_ids) == EXPECTED_N_GALLERY_FULL
    assert len(heldout_gallery_ids_target) == EXPECTED_N_GALLERY_HELDOUT
    assert set(heldout_gallery_ids_target).issubset(set(full_gallery_ids))
    print(f"test identities={len(test_ids)}, full gallery={len(full_gallery_ids)}, "
          f"heldout gallery target={len(heldout_gallery_ids_target)}")

    results = {}
    coverage_checks = {}
    reference_checks = {}

    # ================= NN (v3, quantized) =================
    print("\n--- NN (v3, quantized) ---")
    t0 = time.time()
    gallery_emb_f, gallery_ids_full = m1.embed_baseline(model, gallery_items, is_probe=False)
    probe_emb_f, probe_true_ids = m1.embed_baseline(model, probe_items, is_probe=True)
    assert len(probe_true_ids) == EXPECTED_N_PROBES
    gallery_emb_q = model.quantize(torch.from_numpy(gallery_emb_f)).numpy()
    probe_emb_q = model.quantize(torch.from_numpy(probe_emb_f)).numpy()

    nn_full_metrics, _ = m1.rank_metrics(gallery_emb_q, gallery_ids_full, probe_emb_q, probe_true_ids)
    nn_ref = json.loads(m1._DEDUP_REF_PATH.read_text())["quantized_deduplicated"]
    reference_checks["NN (v3, quantized)"] = check_reference_match(
        "NN (v3, quantized)", nn_full_metrics, nn_ref,
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank"))
    print(f"  full-1998 sanity check passed. ({time.time()-t0:.1f}s)")

    heldout_idx = [i for i, ident in enumerate(gallery_ids_full) if ident in set(heldout_gallery_ids_target)]
    gallery_emb_q_299 = gallery_emb_q[heldout_idx]
    gallery_ids_299 = [gallery_ids_full[i] for i in heldout_idx]
    assert len(gallery_ids_299) == EXPECTED_N_GALLERY_HELDOUT
    coverage_checks["NN (v3, quantized)"] = verify_probe_coverage("NN (v3, quantized)", gallery_ids_299, probe_true_ids)

    nn_heldout_metrics, _ = m1.rank_metrics(gallery_emb_q_299, gallery_ids_299, probe_emb_q, probe_true_ids)
    print(f"  heldout-299 metrics: R@10={nn_heldout_metrics['recall_at_10']:.4f} "
          f"R@50={nn_heldout_metrics['recall_at_50']:.4f}")
    results["NN (v3, quantized)"] = {"full_1998_recomputed": nn_full_metrics,
                                      "full_1998_stored_reference": nn_ref,
                                      "heldout_299": nn_heldout_metrics}

    # ================= Hand-designed features (DFT/AC/RL-C9/RL-C6) =================
    print("\n--- Hand-designed features (casia-codes-2d) ---")
    t0 = time.time()
    gallery_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for identity in full_gallery_ids:
        d = m3._CASIA_2D_DIR / identity
        g_t, g_m = m3.load_template(d, "1"), m3.load_mask_2d(d, "1")
        feats = m2.extract_features_maskaware(g_t, g_m)
        for name in HAND_FEATURE_NAMES:
            gallery_vecs[name].append(feats[name])

    probe_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    hand_probe_true_ids = []
    n_skipped = 0
    for identity in test_ids:
        d = m3._CASIA_2D_DIR / identity
        for stem in m3.available_probe_stems_2d(d):
            if (identity, stem) in m3.DUPLICATE_PROBES:
                n_skipped += 1
                continue
            p_t, p_m = m3.load_template(d, stem), m3.load_mask_2d(d, stem)
            feats = m2.extract_features_maskaware(p_t, p_m)
            hand_probe_true_ids.append(identity)
            for name in HAND_FEATURE_NAMES:
                probe_vecs[name].append(feats[name])
    assert n_skipped == 5
    assert len(hand_probe_true_ids) == EXPECTED_N_PROBES
    print(f"Extracted {len(full_gallery_ids)} galleries, {len(hand_probe_true_ids)} probes "
          f"({time.time()-t0:.1f}s).")

    gallery_idx_full = {gid: i for i, gid in enumerate(full_gallery_ids)}
    true_col_full = np.array([gallery_idx_full[pid] for pid in hand_probe_true_ids])
    row_idx = np.arange(len(hand_probe_true_ids))
    heldout_set = set(heldout_gallery_ids_target)
    heldout_pos_in_full = [i for i, gid in enumerate(full_gallery_ids) if gid in heldout_set]

    for name in HAND_FEATURE_NAMES:
        gallery_full = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])

        D_full = m1.ssd_matrix(gallery_full, probe)
        ranks_full = dense_ranks(D_full)[row_idx, true_col_full] + 1
        full_metrics = metrics_from_ranks(ranks_full, len(full_gallery_ids), len(hand_probe_true_ids))
        reference_checks[name] = check_reference_match(
            name, full_metrics, hand_ref[name],
            ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank"))

        gallery_299 = gallery_full[heldout_pos_in_full]
        gallery_ids_299_hand = [full_gallery_ids[i] for i in heldout_pos_in_full]
        assert len(gallery_ids_299_hand) == EXPECTED_N_GALLERY_HELDOUT
        coverage_checks[name] = verify_probe_coverage(name, gallery_ids_299_hand, hand_probe_true_ids)

        gallery_idx_299 = {gid: i for i, gid in enumerate(gallery_ids_299_hand)}
        true_col_299 = np.array([gallery_idx_299[pid] for pid in hand_probe_true_ids])
        D_299 = m1.ssd_matrix(gallery_299, probe)
        ranks_299 = dense_ranks(D_299)[row_idx, true_col_299] + 1
        heldout_metrics = metrics_from_ranks(ranks_299, EXPECTED_N_GALLERY_HELDOUT, len(hand_probe_true_ids))

        print(f"  {name}: full R@10={full_metrics['recall_at_10']:.4f} (ref OK) -> "
              f"heldout R@10={heldout_metrics['recall_at_10']:.4f}")
        results[name] = {"full_1998_recomputed": full_metrics,
                          "full_1998_stored_reference": hand_ref[name],
                          "heldout_299": heldout_metrics}

    # ================= Christina's feature (inverted convention only) =================
    print("\n--- Christina's RL feature, inverted convention (casia-codes-christina) ---")
    config = FilterFHEConfig()
    t0 = time.time()
    gallery_feats = {}
    for identity in full_gallery_ids:
        d = m3._CASIA_CHRISTINA_DIR / identity
        gcode, gmask_native = m3.load_christina_pair(d, "1")
        gallery_feats[identity] = compute_template_features(gcode, m3.mask_for_convention(gmask_native, "inverted"))
    c_gallery_ids_full = list(gallery_feats.keys())

    probe_records = []
    skipped = 0
    for identity in test_ids:
        d = m3._CASIA_CHRISTINA_DIR / identity
        for stem in m3.available_probe_stems_christina(d):
            if (identity, stem) in m3.DUPLICATE_PROBES:
                skipped += 1
                continue
            pcode, pmask_native = m3.load_christina_pair(d, stem)
            pfeat = compute_template_features(pcode, m3.mask_for_convention(pmask_native, "inverted"))
            probe_records.append((identity, pfeat))
    assert skipped == 5
    assert len(probe_records) == EXPECTED_N_PROBES
    print(f"  {len(c_gallery_ids_full)} galleries, {len(probe_records)} probes "
          f"(extraction {time.time()-t0:.1f}s)")

    def rank_against(gallery_ids_pool):
        t1 = time.time()
        ranks = np.empty(len(probe_records), dtype=np.int64)
        for i, (true_identity, pfeat) in enumerate(probe_records):
            scores = [(gid, compute_l1_distance_plaintext(pfeat, gallery_feats[gid], config))
                      for gid in gallery_ids_pool]
            scores.sort(key=lambda x: x[1])
            ranked_ids = [gid for gid, _ in scores]
            ranks[i] = ranked_ids.index(true_identity) + 1
        return ranks, time.time() - t1

    ranks_full, dt = rank_against(c_gallery_ids_full)
    christina_full_metrics = metrics_from_ranks(ranks_full, len(c_gallery_ids_full), len(probe_records))
    christina_ref = hand_ref["Christina (inverted)"]
    reference_checks["Christina (inverted)"] = check_reference_match(
        "Christina (inverted)", christina_full_metrics, christina_ref,
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank"))
    print(f"  full-1998 ranking done ({dt:.1f}s), sanity check passed.")

    christina_gallery_ids_299 = [gid for gid in c_gallery_ids_full if gid in heldout_set]
    assert len(christina_gallery_ids_299) == EXPECTED_N_GALLERY_HELDOUT
    coverage_checks["Christina (inverted)"] = verify_probe_coverage(
        "Christina (inverted)", christina_gallery_ids_299, [r[0] for r in probe_records])
    ranks_299, dt2 = rank_against(christina_gallery_ids_299)
    christina_heldout_metrics = metrics_from_ranks(ranks_299, EXPECTED_N_GALLERY_HELDOUT, len(probe_records))
    print(f"  heldout-299 ranking done ({dt2:.1f}s): R@10={christina_heldout_metrics['recall_at_10']:.4f}")

    results["Christina (inverted)"] = {"full_1998_recomputed": christina_full_metrics,
                                        "full_1998_stored_reference": christina_ref,
                                        "heldout_299": christina_heldout_metrics}

    # ================= NN-vs-hand-designed Recall@10 gap =================
    gap = {}
    nn_r10_full = results["NN (v3, quantized)"]["full_1998_stored_reference"]["recall_at_10"]
    nn_r10_299 = results["NN (v3, quantized)"]["heldout_299"]["recall_at_10"]
    for name in HAND_FEATURE_NAMES:
        feat_r10_full = results[name]["full_1998_stored_reference"]["recall_at_10"]
        feat_r10_299 = results[name]["heldout_299"]["recall_at_10"]
        gap[name] = {
            "full_1998": {"nn_recall_at_10": nn_r10_full, "feature_recall_at_10": feat_r10_full,
                           "ratio_nn_over_feature": nn_r10_full / feat_r10_full,
                           "absolute_diff": nn_r10_full - feat_r10_full},
            "heldout_299": {"nn_recall_at_10": nn_r10_299, "feature_recall_at_10": feat_r10_299,
                             "ratio_nn_over_feature": nn_r10_299 / feat_r10_299,
                             "absolute_diff": nn_r10_299 - feat_r10_299},
        }

    # ================= Write outputs =================
    out = {
        "metadata": {
            "purpose": "Fully-held-out gallery audit: restricts the gallery to only the 299 "
                       "test identities (removing the 1699 train+val identities that appear as "
                       "distractors in the standard 1998-identity gallery), to check whether the "
                       "NN's advantage over the hand-designed features persists when neither the "
                       "probes nor the gallery distractors were seen during training. The 299-"
                       "identity gallery is a strictly easier task for every feature (fewer "
                       "distractors), so absolute recall rises for all of them; only the "
                       "NN-vs-hand-designed-feature GAP is the quantity of interest.",
            "gallery_path": "casia-extraction/casia-codes-2d (casia-extraction/casia-codes-christina for Christina's feature)",
            "split_path": str(m1._SPLIT_PATH.relative_to(_ROOT)),
            "split_seed": 0,
            "subject_count": {"full_gallery_identities": EXPECTED_N_GALLERY_FULL,
                              "heldout_gallery_identities": EXPECTED_N_GALLERY_HELDOUT,
                              "test_probe_identities": EXPECTED_N_TEST_IDENTITIES},
            "n_probes": EXPECTED_N_PROBES,
            "duplicate_probes_excluded": ["029_L_3", "230_R_5", "236_R_2", "543_L_2", "703_L_7"],
            "seeds_used": {"split_seed": 0, "note": "no randomness in this audit itself; ranking is deterministic"},
            "reference_files": {
                "nn": str(m1._DEDUP_REF_PATH.relative_to(_ROOT)),
                "hand_designed_and_christina": str(_HAND_FEATURE_REF_PATH.relative_to(_ROOT)),
            },
            "distance_metrics": {"NN/DFT/AC/RL-C9/RL-C6": "SSD", "Christina": "L1 (compute_l1_distance_plaintext)"},
            "k_values": K_VALUES,
            "git_commit": m1.get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "reference_checks": reference_checks,
        "probe_genuine_match_coverage_in_heldout_gallery": coverage_checks,
        "results": results,
        "recall_at_10_gap_nn_vs_hand_designed": gap,
    }
    with open(_OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {_OUT_JSON}")

    md = build_markdown(results, gap, coverage_checks)
    with open(_OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved {_OUT_MD}")


def build_markdown(results: dict, gap: dict, coverage_checks: dict) -> str:
    lines = []
    lines.append("# Fully-held-out gallery audit -- NN v3 vs hand-designed features")
    lines.append("")
    lines.append("Read-only audit. Same 299-test-identity probe set (2488 de-duplicated probes) and "
                 "same feature extraction/distances as `all_features_nn_testsplit_comparison.json` / "
                 "`audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. The "
                 "ONLY change is the gallery: 299 test identities only, instead of the full 1998 "
                 "(train+val+test).")
    lines.append("")
    lines.append("**Framing:** the 299-identity gallery has far fewer distractors than the 1998-"
                 "identity gallery, so it is a strictly easier ranking task -- absolute recall rises "
                 "for every feature at the smaller gallery size, for both the NN and the hand-designed "
                 "features. The hand-designed features have no training exposure either way, so they "
                 "benefit only from the smaller gallery; the NN benefits from the smaller gallery AND "
                 "has no seen-in-training distractors. **The point of interest is therefore whether the "
                 "NN-vs-hand-designed-feature gap in Recall@10 is preserved across the two gallery "
                 "scales -- not the absolute recall values themselves.**")
    lines.append("")

    lines.append("## NN vs hand-designed feature gap, Recall@10 (lead result)")
    lines.append("")
    lines.append("| Feature | R@10 (1998-gallery) | R@10 (299-gallery) | Ratio NN/feature (1998) | "
                 "Ratio NN/feature (299) | Abs. diff (1998) | Abs. diff (299) |")
    lines.append("|---|---|---|---|---|---|---|")
    nn_r10_full = results["NN (v3, quantized)"]["full_1998_stored_reference"]["recall_at_10"]
    nn_r10_299 = results["NN (v3, quantized)"]["heldout_299"]["recall_at_10"]
    lines.append(f"| **NN (v3, quantized)** | {nn_r10_full:.4f} | {nn_r10_299:.4f} | -- | -- | -- | -- |")
    for name, g in gap.items():
        f1998, f299 = g["full_1998"], g["heldout_299"]
        lines.append(f"| {name} | {f1998['feature_recall_at_10']:.4f} | {f299['feature_recall_at_10']:.4f} | "
                     f"{f1998['ratio_nn_over_feature']:.3f}x | {f299['ratio_nn_over_feature']:.3f}x | "
                     f"{f1998['absolute_diff']:+.4f} | {f299['absolute_diff']:+.4f} |")
    lines.append("")
    ratios_full = [g["full_1998"]["ratio_nn_over_feature"] for g in gap.values()]
    ratios_299 = [g["heldout_299"]["ratio_nn_over_feature"] for g in gap.values()]
    diffs_full = [g["full_1998"]["absolute_diff"] for g in gap.values()]
    diffs_299 = [g["heldout_299"]["absolute_diff"] for g in gap.values()]
    lines.append(f"Ratio range across the 4 hand-designed features: {min(ratios_full):.3f}x-{max(ratios_full):.3f}x "
                 f"at the 1998-gallery, {min(ratios_299):.3f}x-{max(ratios_299):.3f}x at the 299-gallery. "
                 f"Absolute-difference range: {min(diffs_full):+.4f} to {max(diffs_full):+.4f} at the "
                 f"1998-gallery, {min(diffs_299):+.4f} to {max(diffs_299):+.4f} at the 299-gallery.")
    lines.append("")

    lines.append("## Full metrics, all features, both gallery scales")
    lines.append("")
    lines.append("| Feature | Gallery | R@10 | R@50 | R@100 | R@300 | MedRank | MeanRank |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        f = r["full_1998_stored_reference"]
        h = r["heldout_299"]
        lines.append(f"| {name} | 1998 | {f['recall_at_10']:.4f} | {f['recall_at_50']:.4f} | "
                     f"{f['recall_at_100']:.4f} | {f['recall_at_300']:.4f} | {f['median_rank']:.1f} | "
                     f"{f['mean_rank']:.2f} |")
        lines.append(f"| {name} | 299 | {h['recall_at_10']:.4f} | {h['recall_at_50']:.4f} | "
                     f"{h['recall_at_100']:.4f} | {h['recall_at_300']:.4f} | {h['median_rank']:.1f} | "
                     f"{h['mean_rank']:.2f} |")
    lines.append("")

    lines.append("## Probe genuine-match coverage in the 299-identity gallery")
    lines.append("")
    lines.append("| Feature | Probes | Missing genuine match | All present |")
    lines.append("|---|---|---|---|")
    for name, c in coverage_checks.items():
        lines.append(f"| {name} | {c['n_probes']} | {c['n_missing_genuine_match']} | {c['all_present']} |")
    lines.append("")

    lines.append("## Methodology notes")
    lines.append("")
    lines.append("- Probe set: identical 2488 de-duplicated probes (299 test identities, stems 2-10, "
                 "excluding the 5 byte-identical duplicate probes) in every row above.")
    lines.append("- Gallery restriction: for every feature, gallery vectors were filtered to the 299 "
                 "test identities' own gallery image (stem \"1\"); no other change to extraction, "
                 "distance, or ranking code.")
    lines.append("- Every feature's 1998-gallery numbers shown above are the stored reference values "
                 "(`audit_nn_v3_dedup_metrics.json` for the NN, `audit_all_features_dedup_testsplit.json` "
                 "for the rest); this script independently recomputed each of them first and required an "
                 "exact match (atol=1e-9) before trusting the corresponding 299-gallery number -- see "
                 "`reference_checks` in the JSON output.")
    lines.append("- Distances: SSD for NN/DFT/AC/RL-C9/RL-C6, L1 (`compute_l1_distance_plaintext`) for "
                 "Christina's feature, matching the matched comparison exactly.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
