#!/usr/bin/env python3
"""Mask-geometry leakage audit for NN v3, part 3 -- extends audit_maskleak2_main.py.

Read-only w.r.t. every existing file (checkpoint, split, and every prior
result), same 299-identity / 1998-gallery / 2488-de-duplicated-probe protocol
and metrics as audit_nn_v3_dedup_metrics.json / audit_all_features_dedup_testsplit.json.

CHECK 7 -- repeats Check 5's genuine-pair mask-dissimilarity stratification
(quartiles of 1-IoU between the two TRUE binary masks) for the hand-designed
features (mask-aware DFT / AC / RL-C9, using the exact same extraction calls
as eval_all_features_nn_testsplit.py / audit_all_features_dedup_testsplit.py),
reported side by side with NN v3 in the SAME strata as
audit_maskleak2_results.json's Check 5 (by_iou_dissimilarity).

The quartile ASSIGNMENT itself is not stored as raw indices in
audit_maskleak2_results.json (only per-quartile summary stats are), so it is
reconstructed here by re-running the identical, deterministic computation
(same gallery/probe loading via audit_maskleak_main.load_common, same iou()
function and quartile split via audit_maskleak2_main) on the same probe set --
then VERIFIED to reproduce audit_maskleak2_results.json's stored per-quartile
NN v3 numbers exactly before anything is reported. If that verification does
not pass exactly, the script stops rather than reporting mismatched strata.

A plain Daugman IrisCode Hamming-distance baseline was checked for on this
split and is NOT already computed anywhere in this repo (only
row_redundancy_check.py's unrelated row-redundancy Hamming-distance
diagnostic exists) -- per instruction it is skipped rather than built, and
this is stated explicitly in the output.
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
_RESULTS = _LEVEL1 / "results"
_FEATURES = _LEVEL1 / "features"

sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_FEATURES))

import audit_maskleak_main as m1     # noqa: E402 -- reused unmodified
import audit_maskleak2_main as m2    # noqa: E402 -- reused unmodified (iou, N_QUARTILES)
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k  # noqa: E402
from ac_features import extract_ac_vector      # noqa: E402
from dft_spectrum import code_magnitude_spectrum  # noqa: E402
from rl_features import extract_rl_vector      # noqa: E402

_OUT_JSON = _RESULTS / "audit_maskleak3_results.json"
_OUT_MD = _RESULTS / "audit_maskleak3_RESULTS.md"
_MASKLEAK2_JSON = _RESULTS / "audit_maskleak2_results.json"
_DEDUP_ALLFEATS_JSON = _RESULTS / "audit_all_features_dedup_testsplit.json"

K_VALUES = m1.K_VALUES
N_QUARTILES = m2.N_QUARTILES
DFT_BINS = np.arange(30, 51)
HAND_FEATURE_NAMES = ("DFT", "AC", "RL-C9")


def extract_features_maskaware(template, mask):
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
    }


def quartile_recall(ranks, chunks):
    out = []
    for qi, idx in enumerate(chunks, start=1):
        r = ranks[idx]
        entry = {"quartile": qi, "n_probes": int(len(idx)), "median_rank": float(np.median(r))}
        for k in K_VALUES:
            entry[f"recall_at_{k}"] = recall_at_k(r, k)
        out.append(entry)
    return out


def q1_q4_summary(quartiles):
    q1, q4 = quartiles[0], quartiles[-1]
    out = {}
    for k in K_VALUES:
        key = f"recall_at_{k}"
        v1, v4 = q1[key], q4[key]
        out[f"{key}_Q1"] = v1
        out[f"{key}_Q4"] = v4
        out[f"{key}_relative_drop_Q1_to_Q4"] = float((v1 - v4) / v1) if v1 != 0 else None
    out["median_rank_Q1"] = q1["median_rank"]
    out["median_rank_Q4"] = q4["median_rank"]
    out["median_rank_Q4_over_Q1_ratio"] = (
        float(q4["median_rank"] / q1["median_rank"]) if q1["median_rank"] != 0 else None
    )
    return out


def main():
    if _OUT_JSON.exists():
        raise FileExistsError(f"{_OUT_JSON} already exists; write a new filename instead.")

    print("=== reusing audit_maskleak_main / audit_maskleak2_main (unmodified) ===", flush=True)
    m1.selftest_ssd_ranking()

    test_ids, full_gallery_ids, gallery_items, probe_items, model = m1.load_common()
    print(f"gallery={len(gallery_items)} probes={len(probe_items)}", flush=True)

    print("\n=== baseline NN v3 (unmodified) embeddings ===", flush=True)
    t0 = time.time()
    gallery_emb_f, gallery_ids = m1.embed_baseline(model, gallery_items, is_probe=False)
    probe_emb_f, probe_true_ids = m1.embed_baseline(model, probe_items, is_probe=True)
    gallery_emb_q = model.quantize(torch.from_numpy(gallery_emb_f)).numpy()
    probe_emb_q = model.quantize(torch.from_numpy(probe_emb_f)).numpy()
    baseline_metrics, D_baseline_q = m1.rank_metrics(gallery_emb_q, gallery_ids, probe_emb_q, probe_true_ids)
    print(f"baseline: {baseline_metrics} ({time.time() - t0:.1f}s)", flush=True)

    ref = json.loads(m1._DEDUP_REF_PATH.read_text())["quantized_deduplicated"]
    mismatches = {
        k: (baseline_metrics[k], ref[k])
        for k in ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
        if not np.isclose(baseline_metrics[k], ref[k], atol=1e-9)
    }
    if mismatches:
        raise RuntimeError(f"Recomputed NN baseline does not match audit_nn_v3_dedup_metrics.json: {mismatches}")
    print("[sanity] recomputed NN baseline matches audit_nn_v3_dedup_metrics.json exactly", flush=True)

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])

    # ---------------- reconstruct the exact IoU quartile assignment ----------------
    print("\n=== reconstructing Check 5's IoU quartile assignment ===", flush=True)
    gallery_masks = np.stack([mm for (_, t, mm) in gallery_items])
    probe_masks = np.stack([mm for (_, s, t, mm) in probe_items])
    n_probe = len(probe_true_ids)
    genuine_iou = np.empty(n_probe)
    for i in range(n_probe):
        genuine_iou[i] = m2.iou(probe_masks[i], gallery_masks[true_col[i]])
    diss_iou = 1.0 - genuine_iou
    order_iou = np.argsort(diss_iou)
    chunks = np.array_split(order_iou, N_QUARTILES)
    print(f"quartile sizes: {[len(c) for c in chunks]}", flush=True)

    ranks_nn = ranks_of_true_identity(D_baseline_q, gallery_ids, probe_true_ids)
    nn_quartiles_recomputed = quartile_recall(ranks_nn, chunks)

    m2_json = json.loads(_MASKLEAK2_JSON.read_text())
    stored_nn_quartiles = m2_json["check5_mask_dissimilarity_stratification"]["by_iou_dissimilarity"]
    strata_mismatches = []
    for recomputed, stored in zip(nn_quartiles_recomputed, stored_nn_quartiles):
        for key in ("n_probes", "median_rank") + tuple(f"recall_at_{k}" for k in K_VALUES):
            if not np.isclose(recomputed[key], stored[key], atol=1e-9):
                strata_mismatches.append((recomputed["quartile"], key, recomputed[key], stored[key]))
    if strata_mismatches:
        raise RuntimeError(
            f"Reconstructed IoU quartile assignment does NOT match audit_maskleak2_results.json "
            f"exactly -- refusing to report stratified hand-feature results on possibly-different "
            f"strata. Mismatches: {strata_mismatches}"
        )
    print("[verified] reconstructed IoU quartile assignment reproduces audit_maskleak2_results.json's "
          "stored per-quartile NN v3 numbers EXACTLY (n_probes, median_rank, recall_at_10/50/100/300, "
          "all 4 quartiles) -- strata confirmed identical.", flush=True)

    # ---------------- hand-designed feature extraction (same 2488-probe dedup set) ----------------
    print("\n=== extracting hand-designed features (DFT/AC/RL-C9) ===", flush=True)
    t0 = time.time()
    gallery_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for (_, t, mm) in gallery_items:
        feats = extract_features_maskaware(t, mm)
        for name in HAND_FEATURE_NAMES:
            gallery_vecs[name].append(feats[name])
    probe_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for (_, s, t, mm) in probe_items:
        feats = extract_features_maskaware(t, mm)
        for name in HAND_FEATURE_NAMES:
            probe_vecs[name].append(feats[name])
    print(f"extracted in {time.time() - t0:.1f}s", flush=True)

    hand_quartiles = {}
    hand_aggregate = {}
    for name in HAND_FEATURE_NAMES:
        gallery_arr = np.stack(gallery_vecs[name])
        probe_arr = np.stack(probe_vecs[name])
        D = ssd_matrix(gallery_arr, probe_arr)
        ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)
        agg = {"n_gallery": len(gallery_ids), "n_probes": len(probe_true_ids),
               "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks))}
        for k in K_VALUES:
            agg[f"recall_at_{k}"] = recall_at_k(ranks, k)
        hand_aggregate[name] = agg
        hand_quartiles[name] = quartile_recall(ranks, chunks)
        print(f"  {name} aggregate: {agg}", flush=True)

    # cross-check aggregate hand-feature numbers against the existing dedup evaluation
    dedup_ref = json.loads(_DEDUP_ALLFEATS_JSON.read_text())["results"]
    hand_mismatches = {}
    for name in HAND_FEATURE_NAMES:
        for key in ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank"):
            if not np.isclose(hand_aggregate[name][key], dedup_ref[name][key], atol=1e-9):
                hand_mismatches[f"{name}.{key}"] = (hand_aggregate[name][key], dedup_ref[name][key])
    if hand_mismatches:
        raise RuntimeError(
            f"Recomputed hand-feature aggregate numbers do not match "
            f"audit_all_features_dedup_testsplit.json: {hand_mismatches}"
        )
    print("[sanity] recomputed DFT/AC/RL-C9 aggregate numbers match "
          "audit_all_features_dedup_testsplit.json exactly", flush=True)

    # ---------------- combined table + Q1-to-Q4 relative drop ----------------
    all_quartiles = {"NN v3": nn_quartiles_recomputed, **hand_quartiles}
    drop_summary = {name: q1_q4_summary(qs) for name, qs in all_quartiles.items()}

    print("\n=== combined table (IoU-dissimilarity quartiles) ===", flush=True)
    for name, qs in all_quartiles.items():
        for e in qs:
            print(f"  {name:<8} Q{e['quartile']}: n={e['n_probes']} R@10={e['recall_at_10']:.4f} "
                  f"R@50={e['recall_at_50']:.4f} R@100={e['recall_at_100']:.4f} "
                  f"R@300={e['recall_at_300']:.4f} medRank={e['median_rank']:.1f}", flush=True)

    # ---------------- write JSON ----------------
    result = {
        "metadata": {
            "purpose": "Check 7: repeats audit_maskleak2's Check 5 genuine-pair IoU-dissimilarity "
                       "stratification for the hand-designed features (mask-aware DFT/AC/RL-C9), "
                       "reported alongside NN v3 in the identical strata.",
            "checkpoint_path": str(m1._CKPT_PATH.relative_to(_ROOT)),
            "split_path": str(m1._SPLIT_PATH.relative_to(_ROOT)),
            "reference_maskleak2_check5": str(_MASKLEAK2_JSON.relative_to(_ROOT)),
            "reference_dedup_allfeatures": str(_DEDUP_ALLFEATS_JSON.relative_to(_ROOT)),
            "n_gallery": m1.EXPECTED_N_GALLERY,
            "n_probes": m1.EXPECTED_N_PROBES,
            "n_quartiles": N_QUARTILES,
            "quartile_definition": "identical to audit_maskleak2_results.json's Check 5: genuine pairs "
                                   "sorted ascending by (1 - IoU) of the two true binary masks, split "
                                   f"into {N_QUARTILES} equal-sized chunks of {n_probe // N_QUARTILES} "
                                   "probes each; Q1=most similar masks, Q4=most dissimilar.",
            "strata_verification": "PASSED -- reconstructed quartile assignment reproduces "
                                   "audit_maskleak2_results.json's stored per-quartile NN v3 numbers "
                                   "(n_probes, median_rank, recall_at_10/50/100/300) exactly, for all "
                                   "4 quartiles.",
            "hamming_daugman_baseline": "NOT included: a plain Daugman IrisCode Hamming-distance "
                                        "identification baseline on this exact split is not already "
                                        "computed anywhere in this repo (only row_redundancy_check.py's "
                                        "unrelated row-redundancy Hamming-distance diagnostic exists), "
                                        "and per instruction it was skipped rather than built.",
            "git_commit": m1.get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "nn_v3_baseline_aggregate": baseline_metrics,
        "hand_feature_aggregate": hand_aggregate,
        "hand_feature_aggregate_matches_dedup_reference": True,
        "quartiles_by_feature": all_quartiles,
        "q1_to_q4_summary_by_feature": drop_summary,
    }
    with open(_OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {_OUT_JSON}", flush=True)

    md = build_markdown(baseline_metrics, hand_aggregate, all_quartiles, drop_summary)
    with open(_OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved {_OUT_MD}", flush=True)


def build_markdown(baseline_metrics, hand_aggregate, all_quartiles, drop_summary):
    lines = []
    lines.append("# Mask-geometry leakage audit, part 3 -- CircularConvEncoder v3 vs hand-designed features")
    lines.append("")
    lines.append("Read-only, extends `audit_maskleak2_RESULTS.md`'s Check 5. Same 299-test-identity / "
                 "1998-gallery / 2488-de-duplicated-probe protocol and metrics as "
                 "`audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. "
                 "Numbers only -- no conclusions beyond them.")
    lines.append("")
    lines.append("**Strata verification:** the IoU quartile assignment used below was reconstructed "
                 "(not stored as raw indices in `audit_maskleak2_results.json`) by re-running the "
                 "identical deterministic computation, then verified to reproduce that file's stored "
                 "per-quartile NN v3 numbers (n_probes, median rank, R@10/50/100/300) exactly, for all "
                 "4 quartiles, before anything below was computed. **PASSED.**")
    lines.append("")
    lines.append("**Daugman/Hamming baseline:** not included. A plain IrisCode Hamming-distance "
                 "identification baseline on this exact split is not already computed anywhere in "
                 "this repo, and per instruction was skipped rather than built.")
    lines.append("")

    lines.append("## Combined table -- quartile by quartile (Q1 = most similar masks, Q4 = most dissimilar)")
    lines.append("")
    lines.append("| Feature | Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, qs in all_quartiles.items():
        for e in qs:
            lines.append(f"| {name} | Q{e['quartile']} | {e['n_probes']} | {e['recall_at_10']:.4f} | "
                         f"{e['recall_at_50']:.4f} | {e['recall_at_100']:.4f} | {e['recall_at_300']:.4f} | "
                         f"{e['median_rank']:.1f} |")
    lines.append("")

    lines.append("## Q1-to-Q4 relative drop, per feature")
    lines.append("")
    lines.append("Relative drop = (R@K[Q1] - R@K[Q4]) / R@K[Q1]. Median rank reported as raw Q1/Q4 "
                 "values plus the Q4/Q1 ratio (higher rank = worse, so this is a increase factor, not "
                 "a \"drop\").")
    lines.append("")
    lines.append("| Feature | R@10 drop | R@50 drop | R@100 drop | R@300 drop | MedRank Q1 | MedRank Q4 | Q4/Q1 ratio |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, d in drop_summary.items():
        lines.append(
            f"| {name} | {d['recall_at_10_relative_drop_Q1_to_Q4']:.4f} | "
            f"{d['recall_at_50_relative_drop_Q1_to_Q4']:.4f} | "
            f"{d['recall_at_100_relative_drop_Q1_to_Q4']:.4f} | "
            f"{d['recall_at_300_relative_drop_Q1_to_Q4']:.4f} | "
            f"{d['median_rank_Q1']:.1f} | {d['median_rank_Q4']:.1f} | "
            f"{d['median_rank_Q4_over_Q1_ratio']:.2f} |"
        )
    lines.append("")

    lines.append("## Aggregate (unstratified) hand-feature numbers, for reference")
    lines.append("")
    lines.append("| Feature | R@10 | R@50 | R@100 | R@300 | MedRank | MeanRank |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(f"| NN v3 | {baseline_metrics['recall_at_10']:.4f} | {baseline_metrics['recall_at_50']:.4f} | "
                 f"{baseline_metrics['recall_at_100']:.4f} | {baseline_metrics['recall_at_300']:.4f} | "
                 f"{baseline_metrics['median_rank']:.1f} | {baseline_metrics['mean_rank']:.2f} |")
    for name, a in hand_aggregate.items():
        lines.append(f"| {name} | {a['recall_at_10']:.4f} | {a['recall_at_50']:.4f} | "
                     f"{a['recall_at_100']:.4f} | {a['recall_at_300']:.4f} | "
                     f"{a['median_rank']:.1f} | {a['mean_rank']:.2f} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
