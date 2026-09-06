#!/usr/bin/env python3
"""Representation-overlap audit: does the learned NN embedding (v3, quantized) encode
information linearly reconstructable from the hand-designed features (DFT/AC/RL-C9/
RL-C6), and does it fail on the same probes as they do?

Read-only w.r.t. every existing file, checkpoint, and result. Reuses (imports, never
edits):
  - level1-features/experiments/audit_maskleak_main.py (as m1): load_common() (builds
    the exact 299-test-identity / 1998-gallery / 2488-de-duplicated-probe protocol),
    embed_baseline(), get_git_commit(); and, via m1's own module-level imports,
    nn_train.ssd_matrix / ranks_of_true_identity / recall_at_k and
    nn_model.CircularConvEncoder (loaded inside load_common()).
  - level1-features/features/ac_features.extract_ac_vector, dft_spectrum
    .code_magnitude_spectrum, rl_features.extract_rl_vector -- same mask-aware
    extraction convention (DFT bins 30-50, AC concat, RL-C9, RL-C6) as
    eval_all_features_nn_testsplit.py / audit_all_features_dedup_testsplit.py.

Protocol: identical split/gallery/de-duplication as
all_features_nn_testsplit_comparison.json / audit_nn_v3_dedup_metrics.json /
audit_all_features_dedup_testsplit.json. Both are used as sanity-check references
(recomputed baseline metrics here must match them exactly, atol=1e-9) before any new
analysis is trusted.

No sklearn in this environment -- CCA and linear regression are implemented from
scratch with numpy/scipy only:
  - manual_cca(): classic covariance-based CCA. Cxx/Cyy are inverted via an
    eigendecomposition-based pseudo-inverse square root with a relative eigenvalue
    cutoff (rcond=1e-10 x largest eigenvalue) -- NOT an explicit ridge/regularized
    CCA; this cutoff is the only thing standing between "invert" and "blow up" when
    Cxx/Cyy is near-singular, so cond(Cxx)/cond(Cyy) and the effective rank actually
    used are reported alongside every canonical-correlation number.
  - linear_regression_r2(): plain OLS (no ridge) via np.linalg.lstsq(rcond=None),
    predicting all 64 NN embedding dims from the feature; R^2 per dim, averaged.
    cond(design matrix) is reported alongside.
  - selftest_cca(): run once per feature BEFORE its real CCA number is trusted --
    (a) CCA(X, X) top canonical correlation must be ~= 1.0; (b) CCA(X, a random
    Gaussian matrix of the same n and the NN's column count) top canonical
    correlation must be low. If either fails for any feature, a *_FLAGGED.json is
    written and the script stops before producing final output (same convention as
    audit_maskleak_main.py's Check-1 plausibility gate).

Part 1 (representation overlap, linear only) is computed over the UNION of gallery
(1998, stem "1") + de-duplicated probe (2488) vectors = 4486 total, each image used
exactly once -- see metadata["cca_regression_sample_set"] for the explicit statement
and rationale (this is the full available vector pool on this protocol, and gives the
best sample-to-dimension ratio for the highest-dimensional feature, RL-C9 at 576-dim).

Part 2 (error correlation) is computed on the 2488-probe / 1998-gallery eval protocol
itself (per-probe genuine-match rank vectors), matching the identification benchmark
exactly.

Do not draw conclusions beyond what the numbers support -- this script reports numbers
only; build_markdown() states each result plainly with no interpretation beyond the
finding's own definition.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CASIA_2D_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS = _LEVEL1 / "results"
_DEDUP_REF_PATH = _RESULTS / "audit_nn_v3_dedup_metrics.json"
_HAND_FEATURE_REF_PATH = _RESULTS / "audit_all_features_dedup_testsplit.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

import audit_maskleak_main as m1  # noqa: E402 -- reused unmodified
from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402

_OUT_JSON = _RESULTS / "audit_embcompare_results.json"
_OUT_MD = _RESULTS / "audit_embcompare_RESULTS.md"
_OUT_FLAGGED = _RESULTS / "audit_embcompare_cca_selfcheck_FLAGGED.json"

DFT_BINS = np.arange(30, 51)
HAND_FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
K_VALUES_PART2 = [10, 50]
NN_EMBEDDING_DIM = 64

CCA_RCOND = 1e-10
CCA_SELFCHECK_SELF_MIN = 0.999
CCA_SELFCHECK_RANDOM_MAX = 0.5
CCA_SELFCHECK_SEED = 20260831
CONDITION_NUMBER_FLAG_THRESHOLD = 1e8


# ---------------------------------------------------------------------------
# Hand-designed feature extraction (same convention as
# audit_all_features_dedup_testsplit.py / eval_all_features_nn_testsplit.py)
# ---------------------------------------------------------------------------

def extract_features_maskaware(template: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
        "RL-C6": extract_rl_vector(template, C=6, mask=mask).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Manual CCA (no sklearn available) + self-checks + conditioning report
# ---------------------------------------------------------------------------

def _center(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu


def _pinv_sqrt_eigh(C: np.ndarray, rcond: float):
    """Eigen-decomposition-based pseudo-inverse square root of a symmetric PSD
    matrix, with a relative eigenvalue cutoff. No ridge term is added -- this
    cutoff (drop eigenvalues < rcond * largest eigenvalue) is the sole guard
    against blow-up when C is near-singular, so effective rank is returned
    alongside for transparency."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 0, None)
    max_eig = eigvals.max() if eigvals.size else 0.0
    thresh = rcond * max_eig
    keep = eigvals > thresh
    inv_sqrt_vals = np.zeros_like(eigvals)
    inv_sqrt_vals[keep] = 1.0 / np.sqrt(eigvals[keep])
    inv_sqrt = eigvecs @ np.diag(inv_sqrt_vals) @ eigvecs.T
    return inv_sqrt, int(keep.sum()), int(eigvals.size)


def manual_cca(X: np.ndarray, Y: np.ndarray, rcond: float = CCA_RCOND) -> dict:
    Xc, _ = _center(X)
    Yc, _ = _center(Y)
    n = Xc.shape[0]
    Cxx = (Xc.T @ Xc) / (n - 1)
    Cyy = (Yc.T @ Yc) / (n - 1)
    Cxy = (Xc.T @ Yc) / (n - 1)

    cond_xx = float(np.linalg.cond(Cxx))
    cond_yy = float(np.linalg.cond(Cyy))

    Cxx_is, rank_xx, dim_xx = _pinv_sqrt_eigh(Cxx, rcond)
    Cyy_is, rank_yy, dim_yy = _pinv_sqrt_eigh(Cyy, rcond)

    M = Cxx_is @ Cxy @ Cyy_is
    svals = np.linalg.svd(M, compute_uv=False)
    canonical_correlations = np.clip(svals, 0.0, 1.0)

    return {
        "canonical_correlations": canonical_correlations,
        "cond_Cxx": cond_xx,
        "cond_Cyy": cond_yy,
        "near_singular_Cxx": cond_xx > CONDITION_NUMBER_FLAG_THRESHOLD,
        "near_singular_Cyy": cond_yy > CONDITION_NUMBER_FLAG_THRESHOLD,
        "effective_rank_Cxx": rank_xx,
        "dim_Cxx": dim_xx,
        "effective_rank_Cyy": rank_yy,
        "dim_Cyy": dim_yy,
        "rcond": rcond,
    }


def selftest_cca(X: np.ndarray, feature_name: str, rng: np.random.Generator) -> dict:
    """Must pass before manual_cca(X, nn_embeddings) is trusted for this feature."""
    self_res = manual_cca(X, X.copy())
    top_self = float(self_res["canonical_correlations"][0])
    passed_self = top_self >= CCA_SELFCHECK_SELF_MIN

    R = rng.normal(size=(X.shape[0], NN_EMBEDDING_DIM))
    rand_res = manual_cca(X, R)
    top_random = float(rand_res["canonical_correlations"][0])
    passed_random = top_random < CCA_SELFCHECK_RANDOM_MAX

    return {
        "feature": feature_name,
        "self_pairing_top_corr": top_self,
        "self_pairing_threshold": CCA_SELFCHECK_SELF_MIN,
        "self_pairing_passed": bool(passed_self),
        "random_pairing_top_corr": top_random,
        "random_pairing_threshold": CCA_SELFCHECK_RANDOM_MAX,
        "random_pairing_passed": bool(passed_random),
        "random_pairing_note": "heuristic threshold -- both quantities are reported "
                                "regardless of pass/fail so the reader can judge.",
        "random_seed": CCA_SELFCHECK_SEED,
        "overall_passed": bool(passed_self and passed_random),
    }


# ---------------------------------------------------------------------------
# Linear regression (OLS, no regularization) + conditioning report
# ---------------------------------------------------------------------------

def linear_regression_r2(X: np.ndarray, Y: np.ndarray) -> dict:
    """Plain OLS with intercept via np.linalg.lstsq(rcond=None) -- no ridge term.
    R^2 computed per output (NN) dimension, then averaged."""
    n = X.shape[0]
    design = np.hstack([np.ones((n, 1)), X])
    cond_design = float(np.linalg.cond(design))

    coefs, _, rank, _ = np.linalg.lstsq(design, Y, rcond=None)
    Y_hat = design @ coefs
    ss_res = ((Y - Y_hat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    ss_tot_safe = np.where(ss_tot == 0, np.nan, ss_tot)
    r2_per_dim = 1.0 - ss_res / ss_tot_safe

    return {
        "r2_per_dim_mean": float(np.nanmean(r2_per_dim)),
        "r2_per_dim_min": float(np.nanmin(r2_per_dim)),
        "r2_per_dim_max": float(np.nanmax(r2_per_dim)),
        "cond_design_matrix": cond_design,
        "near_singular_design_matrix": cond_design > CONDITION_NUMBER_FLAG_THRESHOLD,
        "design_matrix_rank": int(rank),
        "design_matrix_dim": int(design.shape[1]),
        "n_samples": int(n),
    }


# ---------------------------------------------------------------------------
# Part 2: error correlation helpers
# ---------------------------------------------------------------------------

def failure_indices(ranks: np.ndarray, k: int) -> set:
    return set(np.where(ranks > k)[0].tolist())


def jaccard(a: set, b: set):
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_markdown(meta, selftests, part1, part2) -> str:
    lines = []
    lines.append("# Embedding vs. hand-designed-feature representation comparison -- NN v3")
    lines.append("")
    lines.append("Read-only audit. Same 299-test-identity / 1998-gallery / 2488-de-duplicated-probe "
                  "protocol as `all_features_nn_testsplit_comparison.json` / "
                  "`audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. "
                  "Numbers only -- no interpretation beyond each result's own definition.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- NN embeddings: quantized 64-dim (`CircularConvEncoder` v3), recomputed baseline "
                  f"verified to match `audit_nn_v3_dedup_metrics.json` (`quantized_deduplicated`) exactly.")
    lines.append(f"- Hand-designed feature aggregate metrics recomputed here match "
                  f"`audit_all_features_dedup_testsplit.json` exactly for all four features "
                  f"(atol=1e-9): {selftests['hand_feature_aggregates_match_reference']}")
    lines.append(f"- **Part 1 sample set:** {meta['cca_regression_sample_set']}")
    lines.append(f"- **Part 2 sample set:** the 2488 de-duplicated test probes ranked against the "
                  "1998-identity gallery (identical to the matched-comparison benchmark).")
    lines.append(f"- No sklearn in this environment; CCA and linear regression are hand-implemented "
                  "(numpy/scipy only) -- see script docstring for exact method.")
    lines.append(f"- CCA whitening uses NO explicit ridge regularization -- Cxx/Cyy are inverted via "
                  f"an eigenvalue pseudo-inverse with a relative cutoff (rcond={CCA_RCOND:.0e}). "
                  f"Regression is plain OLS via `np.linalg.lstsq(rcond=None)`, also unregularized. "
                  f"Condition numbers are reported for every matrix inverted/whitened; a condition "
                  f"number > {CONDITION_NUMBER_FLAG_THRESHOLD:.0e} is flagged as near-singular.")
    lines.append("")

    lines.append("## CCA self-checks (must pass before Part 1 CCA numbers are trusted)")
    lines.append("")
    lines.append("| Feature | Self-pairing top corr (>= 0.999?) | Random-pairing top corr (< 0.5?) | Passed |")
    lines.append("|---|---|---|---|")
    for st in selftests["cca_selfchecks"]:
        lines.append(f"| {st['feature']} | {st['self_pairing_top_corr']:.6f} "
                      f"({'pass' if st['self_pairing_passed'] else 'FAIL'}) | "
                      f"{st['random_pairing_top_corr']:.4f} "
                      f"({'pass' if st['random_pairing_passed'] else 'FAIL'}) | "
                      f"{'PASSED' if st['overall_passed'] else 'FAILED'} |")
    lines.append("")
    lines.append(f"All self-checks passed: **{selftests['all_cca_selfchecks_passed']}**. "
                 "Random-seed for the random-pairing self-check: "
                 f"{CCA_SELFCHECK_SEED}.")
    lines.append("")

    lines.append("## Part 1 -- Representation overlap (linear only)")
    lines.append("")
    lines.append("| Feature | Dim | Top-3 canonical corr | cond(Cxx) | eff.rank(Cxx)/dim | "
                 "cond(Cyy) | eff.rank(Cyy)/dim | Regression R^2 (mean) | cond(design) | "
                 "design rank/dim |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name in HAND_FEATURE_NAMES:
        cca = part1[name]["cca"]
        reg = part1[name]["regression"]
        top3 = ", ".join(f"{c:.4f}" for c in cca["canonical_correlations"][:3])
        lines.append(
            f"| {name} | {part1[name]['dim']} | {top3} | "
            f"{cca['cond_Cxx']:.3e}{' (near-singular)' if cca['near_singular_Cxx'] else ''} | "
            f"{cca['effective_rank_Cxx']}/{cca['dim_Cxx']} | "
            f"{cca['cond_Cyy']:.3e}{' (near-singular)' if cca['near_singular_Cyy'] else ''} | "
            f"{cca['effective_rank_Cyy']}/{cca['dim_Cyy']} | "
            f"{reg['r2_per_dim_mean']:.4f} | "
            f"{reg['cond_design_matrix']:.3e}{' (near-singular)' if reg['near_singular_design_matrix'] else ''} | "
            f"{reg['design_matrix_rank']}/{reg['design_matrix_dim']} |"
        )
    lines.append("")
    lines.append("Regression R^2 range (min/max across the 64 NN dims), for reference:")
    lines.append("")
    lines.append("| Feature | R^2 min | R^2 max |")
    lines.append("|---|---|---|")
    for name in HAND_FEATURE_NAMES:
        reg = part1[name]["regression"]
        lines.append(f"| {name} | {reg['r2_per_dim_min']:.4f} | {reg['r2_per_dim_max']:.4f} |")
    lines.append("")

    lines.append("## Part 2 -- Error correlation (2488-probe eval protocol)")
    lines.append("")
    lines.append("| Feature | Spearman rho | Spearman p | Jaccard@10 | Jaccard@50 | "
                 "NN-fail-count@10 | Feature-fail-count@10 | NN rescues feature@10 | "
                 "Feature rescues NN@10 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name in HAND_FEATURE_NAMES:
        p2 = part2[name]
        lines.append(
            f"| {name} | {p2['spearman_rho']:.4f} | {p2['spearman_p']:.2e} | "
            f"{p2['jaccard_k10']:.4f} | {p2['jaccard_k50']:.4f} | "
            f"{p2['nn_failure_count_k10']} | {p2['feature_failure_count_k10']} | "
            f"{p2['rescue_nn_over_feature_k10']} | {p2['rescue_feature_over_nn_k10']} |"
        )
    lines.append("")
    lines.append("| Feature | Jaccard@50 detail: NN-fail-count@50 | Feature-fail-count@50 |")
    lines.append("|---|---|---|")
    for name in HAND_FEATURE_NAMES:
        p2 = part2[name]
        lines.append(f"| {name} | {p2['nn_failure_count_k50']} | {p2['feature_failure_count_k50']} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if _OUT_JSON.exists():
        raise FileExistsError(f"{_OUT_JSON} already exists; write a new filename instead.")

    print("=== loading split, checkpoint, raw templates/masks (via audit_maskleak_main.load_common) ===",
          flush=True)
    test_ids, full_gallery_ids, gallery_items, probe_items, model = m1.load_common()
    print(f"gallery={len(gallery_items)} probes={len(probe_items)}", flush=True)

    # ---------------- NN embeddings (quantized), baseline sanity check ----------------
    print("\n=== NN embeddings (quantized) ===", flush=True)
    gallery_emb_f, gallery_ids = m1.embed_baseline(model, gallery_items, is_probe=False)
    probe_emb_f, probe_true_ids = m1.embed_baseline(model, probe_items, is_probe=True)
    gallery_emb_q = model.quantize(torch.from_numpy(gallery_emb_f)).numpy().astype(np.float64)
    probe_emb_q = model.quantize(torch.from_numpy(probe_emb_f)).numpy().astype(np.float64)

    D_nn = m1.ssd_matrix(gallery_emb_q, probe_emb_q)
    ranks_nn = m1.ranks_of_true_identity(D_nn, gallery_ids, probe_true_ids)
    nn_metrics = {
        "n_gallery": len(gallery_ids), "n_probes": len(probe_true_ids),
        "median_rank": float(np.median(ranks_nn)), "mean_rank": float(np.mean(ranks_nn)),
        **{f"recall_at_{k}": m1.recall_at_k(ranks_nn, k) for k in (10, 50, 100, 300)},
    }
    ref = json.loads(_DEDUP_REF_PATH.read_text())["quantized_deduplicated"]
    mismatches = {
        k: (nn_metrics[k], ref[k]) for k in
        ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
        if not np.isclose(nn_metrics[k], ref[k], atol=1e-9)
    }
    if mismatches:
        raise RuntimeError(f"Recomputed NN baseline does not match audit_nn_v3_dedup_metrics.json: {mismatches}")
    print("[sanity] recomputed NN baseline matches audit_nn_v3_dedup_metrics.json exactly", flush=True)

    # ---------------- Hand-designed features + sanity check ----------------
    print("\n=== extracting hand-designed features (DFT/AC/RL-C9/RL-C6) ===", flush=True)
    t0 = time.time()
    gallery_feats = {name: [] for name in HAND_FEATURE_NAMES}
    for (identity, t, mask) in gallery_items:
        feats = extract_features_maskaware(t, mask)
        for name in HAND_FEATURE_NAMES:
            gallery_feats[name].append(feats[name])
    probe_feats = {name: [] for name in HAND_FEATURE_NAMES}
    for (identity, stem, t, mask) in probe_items:
        feats = extract_features_maskaware(t, mask)
        for name in HAND_FEATURE_NAMES:
            probe_feats[name].append(feats[name])
    for name in HAND_FEATURE_NAMES:
        gallery_feats[name] = np.stack(gallery_feats[name]).astype(np.float64)
        probe_feats[name] = np.stack(probe_feats[name]).astype(np.float64)
    print(f"extracted in {time.time() - t0:.1f}s. dims: "
          f"{ {name: gallery_feats[name].shape[1] for name in HAND_FEATURE_NAMES} }", flush=True)

    hand_ref = json.loads(_HAND_FEATURE_REF_PATH.read_text())["results"]
    hand_feature_ranks = {}
    hand_matches_ref = {}
    for name in HAND_FEATURE_NAMES:
        D_feat = m1.ssd_matrix(gallery_feats[name], probe_feats[name])
        ranks_feat = m1.ranks_of_true_identity(D_feat, gallery_ids, probe_true_ids)
        hand_feature_ranks[name] = ranks_feat
        recomputed = {
            "median_rank": float(np.median(ranks_feat)), "mean_rank": float(np.mean(ranks_feat)),
            **{f"recall_at_{k}": m1.recall_at_k(ranks_feat, k) for k in (10, 50, 100, 300)},
        }
        ref_entry = hand_ref[name]
        ok = all(
            np.isclose(recomputed[k], ref_entry[k], atol=1e-9)
            for k in ("median_rank", "mean_rank", "recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300")
        )
        hand_matches_ref[name] = bool(ok)
        if not ok:
            raise RuntimeError(f"Recomputed {name} does not match audit_all_features_dedup_testsplit.json: "
                               f"{recomputed} vs {ref_entry}")
    print(f"[sanity] recomputed hand-feature aggregates match audit_all_features_dedup_testsplit.json "
          f"exactly for all 4 features: {hand_matches_ref}", flush=True)

    # ---------------- Part 1: representation overlap (CCA + regression) ----------------
    print("\n=== Part 1: CCA + linear regression (union of gallery+probe, 4486 vectors) ===", flush=True)
    union_nn = np.concatenate([gallery_emb_q, probe_emb_q], axis=0)
    union_feats = {name: np.concatenate([gallery_feats[name], probe_feats[name]], axis=0)
                   for name in HAND_FEATURE_NAMES}
    n_union = union_nn.shape[0]
    print(f"union sample count = {n_union}", flush=True)

    cca_selfchecks = []
    part1 = {}
    base_rng = np.random.default_rng(CCA_SELFCHECK_SEED)
    for name in HAND_FEATURE_NAMES:
        X = union_feats[name]
        st = selftest_cca(X, name, base_rng)
        cca_selfchecks.append(st)
        print(f"  [selftest_cca] {name}: self_pairing_top_corr={st['self_pairing_top_corr']:.6f} "
              f"({'PASS' if st['self_pairing_passed'] else 'FAIL'}), "
              f"random_pairing_top_corr={st['random_pairing_top_corr']:.4f} "
              f"({'PASS' if st['random_pairing_passed'] else 'FAIL'})", flush=True)

    all_selfchecks_passed = all(st["overall_passed"] for st in cca_selfchecks)
    if not all_selfchecks_passed:
        flagged = {
            "reason": "One or more CCA self-checks failed (self-pairing top corr < "
                      f"{CCA_SELFCHECK_SELF_MIN} and/or random-pairing top corr >= "
                      f"{CCA_SELFCHECK_RANDOM_MAX}). Stopping before computing/reporting any real "
                      "CCA numbers, per instruction.",
            "cca_selfchecks": cca_selfchecks,
        }
        with open(_OUT_FLAGGED, "w") as f:
            json.dump(flagged, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
        print(f"\n!!! CCA SELF-CHECK FAILED -- see {_OUT_FLAGGED}. Stopping. !!!", flush=True)
        return
    print("[sanity] all CCA self-checks PASSED -- proceeding to real CCA numbers.", flush=True)

    for name in HAND_FEATURE_NAMES:
        X = union_feats[name]
        t0 = time.time()
        cca_res = manual_cca(X, union_nn)
        reg_res = linear_regression_r2(X, union_nn)
        part1[name] = {
            "dim": int(X.shape[1]),
            "cca": {
                "canonical_correlations": cca_res["canonical_correlations"].tolist(),
                "cond_Cxx": cca_res["cond_Cxx"], "cond_Cyy": cca_res["cond_Cyy"],
                "near_singular_Cxx": cca_res["near_singular_Cxx"],
                "near_singular_Cyy": cca_res["near_singular_Cyy"],
                "effective_rank_Cxx": cca_res["effective_rank_Cxx"], "dim_Cxx": cca_res["dim_Cxx"],
                "effective_rank_Cyy": cca_res["effective_rank_Cyy"], "dim_Cyy": cca_res["dim_Cyy"],
                "rcond": cca_res["rcond"],
            },
            "regression": reg_res,
        }
        print(f"  {name}: top-3 canonical corr = {cca_res['canonical_correlations'][:3]}, "
              f"cond(Cxx)={cca_res['cond_Cxx']:.3e}, cond(Cyy)={cca_res['cond_Cyy']:.3e}, "
              f"R^2(mean)={reg_res['r2_per_dim_mean']:.4f}, cond(design)={reg_res['cond_design_matrix']:.3e} "
              f"({time.time() - t0:.1f}s)", flush=True)

    # ---------------- Part 2: error correlation ----------------
    print("\n=== Part 2: error correlation (Spearman, Jaccard, rescue counts) ===", flush=True)
    part2 = {}
    for name in HAND_FEATURE_NAMES:
        ranks_feat = hand_feature_ranks[name]
        rho, p = spearmanr(ranks_nn, ranks_feat)

        entry = {"spearman_rho": float(rho), "spearman_p": float(p)}
        for k in K_VALUES_PART2:
            nn_fail = failure_indices(ranks_nn, k)
            feat_fail = failure_indices(ranks_feat, k)
            entry[f"jaccard_k{k}"] = jaccard(nn_fail, feat_fail)
            entry[f"nn_failure_count_k{k}"] = len(nn_fail)
            entry[f"feature_failure_count_k{k}"] = len(feat_fail)

        nn_top10 = set(np.where(ranks_nn <= 10)[0].tolist())
        feat_top10 = set(np.where(ranks_feat <= 10)[0].tolist())
        entry["rescue_nn_over_feature_k10"] = len(nn_top10 - feat_top10)
        entry["rescue_feature_over_nn_k10"] = len(feat_top10 - nn_top10)

        part2[name] = entry
        print(f"  {name}: spearman_rho={rho:.4f}, jaccard@10={entry['jaccard_k10']:.4f}, "
              f"jaccard@50={entry['jaccard_k50']:.4f}, "
              f"rescue(NN>feat)@10={entry['rescue_nn_over_feature_k10']}, "
              f"rescue(feat>NN)@10={entry['rescue_feature_over_nn_k10']}", flush=True)

    # ---------------- write consolidated JSON ----------------
    cca_sample_set_note = (
        "Union of gallery (1998 identities, stem '1') + de-duplicated probe (2488) vectors "
        "= 4486 total, each image used exactly once. Chosen (rather than probes-only or "
        "gallery-only) to maximize the sample-to-dimension ratio for the highest-dimensional "
        "feature (RL-C9, 576-dim): 4486 samples vs. 1998 (gallery-only, ratio ~3.5x) or "
        "2488 (probes-only, ratio ~4.3x). Part 2 (error correlation) instead uses only the "
        "2488-probe / 1998-gallery eval protocol, since it needs the actual identification ranks."
    )

    result = {
        "metadata": {
            "purpose": "Representation-overlap and error-correlation audit between the learned "
                       "CircularConvEncoder v3 embedding and each hand-designed Level-1 feature "
                       "(DFT, AC, RL-C9, RL-C6), on the identical 299-test-identity / 1998-gallery / "
                       "2488-de-duplicated-probe protocol as all_features_nn_testsplit_comparison.json "
                       "/ audit_nn_v3_dedup_metrics.json / audit_all_features_dedup_testsplit.json.",
            "checkpoint_path": str(m1._CKPT_PATH.relative_to(_ROOT)),
            "split_path": str(m1._SPLIT_PATH.relative_to(_ROOT)),
            "gallery_path": str(_CASIA_2D_DIR.relative_to(_ROOT)),
            "reference_dedup_nn_metrics": str(_DEDUP_REF_PATH.relative_to(_ROOT)),
            "reference_hand_feature_metrics": str(_HAND_FEATURE_REF_PATH.relative_to(_ROOT)),
            "n_test_identities": len(test_ids),
            "n_gallery_identities": len(gallery_ids),
            "n_probes_deduplicated": len(probe_true_ids),
            "n_union_samples_part1": n_union,
            "cca_regression_sample_set": cca_sample_set_note,
            "part2_sample_set": "2488 de-duplicated test probes vs. 1998-identity gallery "
                                "(matched-comparison protocol).",
            "cca_method": "Manual covariance-based CCA (no sklearn available). Cxx/Cyy inverted via "
                          f"eigenvalue pseudo-inverse square root, relative cutoff rcond={CCA_RCOND:.0e} "
                          "x largest eigenvalue -- no explicit ridge regularization added.",
            "regression_method": "Plain OLS with intercept via np.linalg.lstsq(rcond=None) -- no ridge "
                                 "regularization added.",
            "condition_number_flag_threshold": CONDITION_NUMBER_FLAG_THRESHOLD,
            "cca_selfcheck_seeds": {"random_pairing_seed": CCA_SELFCHECK_SEED},
            "hand_feature_dims": {name: int(union_feats[name].shape[1]) for name in HAND_FEATURE_NAMES},
            "nn_embedding_dim": NN_EMBEDDING_DIM,
            "distance_metric": "SSD (embeddings and hand features alike), same as the matched "
                              "identification benchmark.",
            "git_commit": m1.get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "selftests": {
            "nn_baseline_matches_dedup_reference": True,
            "hand_feature_aggregates_match_reference": hand_matches_ref,
            "cca_selfchecks": cca_selfchecks,
            "all_cca_selfchecks_passed": all_selfchecks_passed,
        },
        "part1_representation_overlap": part1,
        "part2_error_correlation": part2,
    }
    with open(_OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {_OUT_JSON}", flush=True)

    md = build_markdown(
        result["metadata"],
        {
            "hand_feature_aggregates_match_reference": hand_matches_ref,
            "cca_selfchecks": cca_selfchecks,
            "all_cca_selfchecks_passed": all_selfchecks_passed,
        },
        part1, part2,
    )
    with open(_OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved {_OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
