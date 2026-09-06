#!/usr/bin/env python3
"""Mask-geometry leakage audit for NN v3, part 2 -- extends audit_maskleak_main.py.

Read-only w.r.t. every existing file (checkpoint, split, and every prior
result), same 299-identity / 1998-gallery / 2488-de-duplicated-probe protocol
and metrics as audit_nn_v3_dedup_metrics.json. Reuses (imports, never edits)
audit_maskleak_main.py's loaders, model, self-tests, and ranking/metric
functions.

CHECK 4 -- constant-mask channel: channel 1 (fed to the conv stack) set to
all-ones for every image; channel 0 (iris code, zeroed at occluded positions
by the TRUE mask) and the pooling-stage mask both stay TRUE. Reuses
forward_with_mask_swap from audit_maskleak_main.py verbatim (just passing an
all-ones tensor as the "swap" mask instead of another identity's mask), and
re-confirms the no-swap equivalence self-check (channel1==true mask matches
model(x) to < 1e-6) before trusting any constant-mask number.

CHECK 5 -- mask-dissimilarity stratification of genuine pairs: quartiles of
(1 - IoU) between the two TRUE binary masks, and (secondary) quartiles of
32-dim row-validity-profile Euclidean distance. No forward pass -- reuses the
baseline embeddings/distances already computed here. Gallery has exactly one
image per identity (stem '1'), so every probe has exactly one genuine gallery
match; there is no multi-genuine-pair case to resolve.

CHECK 6 -- partial correlation (extends Check 3): Pearson/Spearman between
embedding SSD and mask-profile Euclidean distance on impostor pairs,
controlling for (v_A + v_B) and |v_A - v_B| (each image's overall valid-bit
fraction), via residuals of an OLS regression on those two covariates.
Partial Spearman is computed by rank-transforming all four variables first,
then taking the partial Pearson correlation of the ranks (a standard
nonparametric proxy for partial Spearman).
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, rankdata

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_RESULTS = _LEVEL1 / "results"

sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_LEVEL1 / "features"))

import audit_maskleak_main as m1  # noqa: E402 -- reused unmodified
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k  # noqa: E402

_OUT_JSON = _RESULTS / "audit_maskleak2_results.json"
_OUT_MD = _RESULTS / "audit_maskleak2_RESULTS.md"

K_VALUES = m1.K_VALUES
N_QUARTILES = 4


def row_validity(mask):
    return mask.astype(np.float64).mean(axis=1)  # (32,)


def total_validity(mask):
    return float(mask.astype(np.float64).mean())  # scalar, whole-mask valid fraction


def iou(mask_a, mask_b):
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def partial_corr(x, y, z1, z2):
    """Pearson correlation of the residuals of x and y after OLS-regressing
    each on [z1, z2, intercept]."""
    Z = np.column_stack([z1, z2, np.ones_like(z1)])
    coef_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    coef_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resx = x - Z @ coef_x
    resy = y - Z @ coef_y
    return float(np.corrcoef(resx, resy)[0, 1])


def embed_with_constant_mask(model, items, is_probe, batch_size=64):
    """Same call path as audit_maskleak_main.embed_with_swap, but channel 1
    fed to the conv stack is all-ones for every image (not another
    identity's mask). Channel 0 and the pooling mask both stay TRUE."""
    embs, ids = [], []
    code_batch, onesmask_batch, truemask_batch = [], [], []

    def flush():
        nonlocal code_batch, onesmask_batch, truemask_batch
        if not code_batch:
            return
        codes = torch.stack(code_batch)
        ones_ = torch.stack(onesmask_batch)
        trues = torch.stack(truemask_batch)
        with torch.no_grad():
            e = m1.forward_with_mask_swap(model, codes, ones_, trues).numpy()
        embs.append(e)
        code_batch, onesmask_batch, truemask_batch = [], [], []

    for item in items:
        ident, t, mm = (item[0], item[2], item[3]) if is_probe else item
        x = m1.to_input_tensor(t, mm)
        code, true_mask = x[0], x[1]
        code_batch.append(code)
        onesmask_batch.append(torch.ones_like(true_mask))
        truemask_batch.append(true_mask)
        ids.append(ident)
        if len(code_batch) == batch_size:
            flush()
    flush()
    return np.concatenate(embs, axis=0), ids


def _fmt_row5(e):
    return (f"| Q{e['quartile']} | {e['n_probes']} | {e['recall_at_10']:.4f} | {e['recall_at_50']:.4f} | "
            f"{e['recall_at_100']:.4f} | {e['recall_at_300']:.4f} | {e['median_rank']:.1f} | "
            f"{e['mean_genuine_embedding_distance']:.2f} | {e['mean_impostor_embedding_distance']:.2f} |")


def build_markdown(baseline_metrics, check4_metrics, equiv_diff,
                    check5_iou, check5_rowdist,
                    n_pairs, pearson_r, spearman_r, partial_pearson, partial_spearman):
    lines = []
    lines.append("# Mask-geometry leakage audit, part 2 -- CircularConvEncoder v3")
    lines.append("")
    lines.append("Read-only, extends `audit_maskleak_RESULTS.md`. Same 299-test-identity / "
                 "1998-gallery / 2488-de-duplicated-probe protocol and metrics as "
                 "`audit_nn_v3_dedup_metrics.json`. Numbers only -- no conclusions beyond them.")
    lines.append("")
    lines.append("## Reference baseline (unmodified NN v3, quantized)")
    lines.append("")
    lines.append("| | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(m1._fmt_row("NN v3 (unmodified)", baseline_metrics))
    lines.append("")

    lines.append("## Check 4 -- constant-mask (all-ones) channel 1")
    lines.append("")
    lines.append(f"Self-check (channel 1 = true mask reproduces `model(x)`): max abs diff = "
                 f"{equiv_diff:.3e} (< 1e-6 required).")
    lines.append("")
    lines.append("| | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(m1._fmt_row("Constant-mask channel 1", check4_metrics))
    lines.append(m1._fmt_row("Baseline (unmodified)", baseline_metrics))
    lines.append("")

    lines.append("## Check 5 -- mask-dissimilarity stratification of genuine pairs")
    lines.append("")
    lines.append("Gallery has exactly one image per identity (stem '1'), so every probe has exactly "
                 "one genuine gallery match -- no multi-genuine-pair handling was needed. "
                 f"Genuine pairs (n=2488) split into {N_QUARTILES} equal-sized quartiles of dissimilarity "
                 "(Q1 = most similar masks, Q4 = most dissimilar). \"Impostor dist\" is each quartile's "
                 "probes' mean per-probe-average SSD to all non-genuine gallery entries.")
    lines.append("")
    lines.append("### Stratified by (1 - IoU) of the two true binary masks")
    lines.append("")
    lines.append("| Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank | Mean genuine dist | Mean impostor dist |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in check5_iou:
        lines.append(_fmt_row5(e))
    lines.append("")
    lines.append("### Stratified by 32-dim row-validity-profile Euclidean distance")
    lines.append("")
    lines.append("| Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank | Mean genuine dist | Mean impostor dist |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in check5_rowdist:
        lines.append(_fmt_row5(e))
    lines.append("")

    lines.append("## Check 6 -- partial correlation (extends Check 3)")
    lines.append("")
    lines.append(f"n = {n_pairs} impostor pairs (same set as Check 3). Controlling for "
                 "(v_A + v_B) and |v_A - v_B|, where v is each image's overall valid-bit fraction.")
    lines.append("")
    lines.append("| | Pearson r | Spearman rho |")
    lines.append("|---|---|---|")
    lines.append(f"| Unconditional (Check 3 reference) | {pearson_r:.4f} | {spearman_r:.4f} |")
    lines.append(f"| Partial, controlling v_sum & v_diff | {partial_pearson:.4f} | {partial_spearman:.4f} |")
    lines.append("")
    lines.append("Partial Spearman computed by rank-transforming all four variables (embedding distance, "
                 "mask-profile distance, v_sum, v_diff) and then taking the partial Pearson correlation "
                 "of the ranks.")
    lines.append("")

    return "\n".join(lines)


def main():
    if _OUT_JSON.exists():
        raise FileExistsError(f"{_OUT_JSON} already exists; write a new filename instead.")

    print("=== reusing audit_maskleak_main's self-tests, loaders, model (unmodified) ===", flush=True)
    m1.selftest_ssd_ranking()

    test_ids, full_gallery_ids, gallery_items, probe_items, model = m1.load_common()
    print(f"gallery={len(gallery_items)} probes={len(probe_items)}", flush=True)

    sample_inputs = [m1.to_input_tensor(t, mm) for (_, t, mm) in gallery_items[:8]]
    equiv_diff = m1.selftest_forward_equivalence(model, sample_inputs)
    print(f"[selftest] channel1=true-mask equivalence re-confirmed: max abs diff={equiv_diff:.3e}", flush=True)

    print("\n=== baseline (unmodified) embeddings ===", flush=True)
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
        raise RuntimeError(f"Recomputed baseline does not match audit_nn_v3_dedup_metrics.json: {mismatches}")
    print("[sanity] recomputed baseline matches audit_nn_v3_dedup_metrics.json exactly", flush=True)

    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(len(probe_true_ids))

    # ---------------- CHECK 4: constant-mask channel ----------------
    print("\n=== CHECK 4: constant-mask (all-ones) channel-1 ===", flush=True)
    t0 = time.time()
    g_emb_f4, g_ids4 = embed_with_constant_mask(model, gallery_items, is_probe=False)
    p_emb_f4, p_true_ids4 = embed_with_constant_mask(model, probe_items, is_probe=True)
    g_emb_q4 = model.quantize(torch.from_numpy(g_emb_f4)).numpy()
    p_emb_q4 = model.quantize(torch.from_numpy(p_emb_f4)).numpy()
    check4_metrics, _ = m1.rank_metrics(g_emb_q4, g_ids4, p_emb_q4, p_true_ids4)
    print(f"check4 (constant-mask channel) metrics: {check4_metrics} ({time.time() - t0:.1f}s)", flush=True)

    # ---------------- CHECK 5: mask-dissimilarity stratification ----------------
    print("\n=== CHECK 5: mask-dissimilarity stratification (genuine pairs) ===", flush=True)
    ranks = ranks_of_true_identity(D_baseline_q, gallery_ids, probe_true_ids)

    gallery_masks = np.stack([mm for (_, t, mm) in gallery_items])   # (1998,32,512), gallery_ids order
    probe_masks = np.stack([mm for (_, s, t, mm) in probe_items])     # (2488,32,512), probe_true_ids order

    n_probe = len(probe_true_ids)
    genuine_iou = np.empty(n_probe)
    for i in range(n_probe):
        genuine_iou[i] = iou(probe_masks[i], gallery_masks[true_col[i]])
    diss_iou = 1.0 - genuine_iou
    order_iou = np.argsort(diss_iou)

    gallery_rowvec = np.stack([row_validity(mm) for (_, t, mm) in gallery_items])
    probe_rowvec = np.stack([row_validity(mm) for (_, s, t, mm) in probe_items])
    genuine_rowdist = np.linalg.norm(probe_rowvec - gallery_rowvec[true_col], axis=1)
    order_rowdist = np.argsort(genuine_rowdist)

    row_sum = D_baseline_q.sum(axis=1)
    genuine_dist = D_baseline_q[row_idx, true_col]
    per_probe_impostor_mean = (row_sum - genuine_dist) / (D_baseline_q.shape[1] - 1)

    def quartiles_with_distance(order):
        chunks = np.array_split(order, N_QUARTILES)
        out = []
        for qi, idx in enumerate(chunks, start=1):
            r = ranks[idx]
            entry = {
                "quartile": qi,
                "n_probes": int(len(idx)),
                "median_rank": float(np.median(r)),
                "mean_genuine_embedding_distance": float(genuine_dist[idx].mean()),
                "mean_impostor_embedding_distance": float(per_probe_impostor_mean[idx].mean()),
            }
            for k in K_VALUES:
                entry[f"recall_at_{k}"] = recall_at_k(r, k)
            out.append(entry)
        return out

    check5_iou = quartiles_with_distance(order_iou)
    check5_rowdist = quartiles_with_distance(order_rowdist)
    for e in check5_iou:
        print(f"  [IoU]     Q{e['quartile']}: n={e['n_probes']} R@10={e['recall_at_10']:.4f} "
              f"medRank={e['median_rank']:.1f} genDist={e['mean_genuine_embedding_distance']:.2f} "
              f"impDist={e['mean_impostor_embedding_distance']:.2f}", flush=True)
    for e in check5_rowdist:
        print(f"  [RowDist] Q{e['quartile']}: n={e['n_probes']} R@10={e['recall_at_10']:.4f} "
              f"medRank={e['median_rank']:.1f} genDist={e['mean_genuine_embedding_distance']:.2f} "
              f"impDist={e['mean_impostor_embedding_distance']:.2f}", flush=True)

    # ---------------- CHECK 6: partial correlation ----------------
    print("\n=== CHECK 6: partial correlation controlling for total mask validity ===", flush=True)
    D_mask = ssd_matrix(gallery_rowvec, probe_rowvec)
    impostor_mask = np.ones_like(D_baseline_q, dtype=bool)
    impostor_mask[row_idx, true_col] = False

    emb_dist = D_baseline_q[impostor_mask]
    mask_dist = np.sqrt(np.clip(D_mask[impostor_mask], 0, None))
    n_pairs = emb_dist.shape[0]
    print(f"n impostor pairs = {n_pairs}", flush=True)

    v_probe = np.array([total_validity(mm) for (_, s, t, mm) in probe_items])
    v_gallery = np.array([total_validity(mm) for (_, t, mm) in gallery_items])
    V_A = np.broadcast_to(v_probe[:, None], D_baseline_q.shape)
    V_B = np.broadcast_to(v_gallery[None, :], D_baseline_q.shape)
    v_sum = (V_A + V_B)[impostor_mask]
    v_diff = np.abs(V_A - V_B)[impostor_mask]

    t0 = time.time()
    pearson_r, pearson_p = pearsonr(emb_dist, mask_dist)
    spearman_r, spearman_p = spearmanr(emb_dist, mask_dist)
    print(f"[reference, unconditional] pearson r={pearson_r:.4f} (p={pearson_p:.2e}), "
          f"spearman rho={spearman_r:.4f} (p={spearman_p:.2e}) ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    partial_pearson = partial_corr(emb_dist, mask_dist, v_sum, v_diff)
    print(f"partial pearson (controlling v_sum, v_diff) = {partial_pearson:.4f} ({time.time() - t0:.1f}s)",
          flush=True)

    t0 = time.time()
    r_emb, r_mask = rankdata(emb_dist), rankdata(mask_dist)
    r_vsum, r_vdiff = rankdata(v_sum), rankdata(v_diff)
    partial_spearman = partial_corr(r_emb, r_mask, r_vsum, r_vdiff)
    print(f"partial spearman (rank-transform + partial pearson on ranks) = {partial_spearman:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---------------- write consolidated JSON ----------------
    result = {
        "metadata": {
            "purpose": "Part 2 of the mask-geometry leakage audit for CircularConvEncoder v3: "
                       "constant-mask-channel ablation (Check 4), mask-dissimilarity stratification "
                       "of genuine pairs (Check 5), and partial correlation of embedding vs "
                       "mask-profile distance controlling for total mask validity (Check 6, extends "
                       "Check 3 of audit_maskleak_results.json).",
            "checkpoint_path": str(m1._CKPT_PATH.relative_to(_ROOT)),
            "split_path": str(m1._SPLIT_PATH.relative_to(_ROOT)),
            "reference_dedup_metrics": str(m1._DEDUP_REF_PATH.relative_to(_ROOT)),
            "reference_check3_source": "level1-features/results/audit_maskleak_results.json (check3_distance_correlation)",
            "n_gallery": m1.EXPECTED_N_GALLERY,
            "n_probes": m1.EXPECTED_N_PROBES,
            "n_quartiles": N_QUARTILES,
            "distance_metric": "SSD (embeddings), Euclidean (32-dim row-validity mask profile), IoU (binary masks)",
            "git_commit": m1.get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "selftests": {
            "ssd_ranking_toy_check": "PASSED (reused from audit_maskleak_main.py)",
            "forward_with_mask_swap_equivalence_max_abs_diff": equiv_diff,
        },
        "baseline_quantized_metrics": baseline_metrics,
        "baseline_matches_dedup_reference": True,
        "check4_constant_mask_channel": {
            **check4_metrics,
            "baseline_for_comparison": baseline_metrics,
        },
        "check5_mask_dissimilarity_stratification": {
            "genuine_pair_definition": "gallery contains exactly one image per identity (stem '1'), "
                                       "so every probe has exactly one genuine gallery match -- no "
                                       "multi-genuine-pair handling was needed.",
            "n_quartiles": N_QUARTILES,
            "quartile_definition": "genuine pairs sorted ascending by dissimilarity and split into "
                                   f"{N_QUARTILES} equal-sized chunks (n={n_probe}/{N_QUARTILES}="
                                   f"{n_probe // N_QUARTILES} each); Q1=most similar masks, "
                                   f"Q{N_QUARTILES}=most dissimilar.",
            "by_iou_dissimilarity": check5_iou,
            "by_row_validity_profile_distance": check5_rowdist,
        },
        "check6_partial_correlation": {
            "n_impostor_pairs": int(n_pairs),
            "reference_pearson_r": float(pearson_r),
            "reference_pearson_p": float(pearson_p),
            "reference_spearman_rho": float(spearman_r),
            "reference_spearman_p": float(spearman_p),
            "partial_pearson_controlling_v_sum_v_diff": partial_pearson,
            "partial_spearman_controlling_v_sum_v_diff": partial_spearman,
            "covariates": "v_sum = v_probe + v_gallery, v_diff = |v_probe - v_gallery|, where v is "
                         "each image's overall valid-bit fraction (mean over the full 32x512 mask)",
            "method_note": "Partial correlation = Pearson correlation of the residuals of each "
                          "variable after OLS regression on [v_sum, v_diff, intercept]. Partial "
                          "Spearman computed by rank-transforming all four variables first, then "
                          "taking the partial Pearson correlation of the ranks.",
        },
    }
    with open(_OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {_OUT_JSON}", flush=True)

    md = build_markdown(baseline_metrics, check4_metrics, equiv_diff,
                         check5_iou, check5_rowdist,
                         n_pairs, pearson_r, spearman_r, partial_pearson, partial_spearman)
    with open(_OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved {_OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
