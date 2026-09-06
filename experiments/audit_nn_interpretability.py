#!/usr/bin/env python3
"""Representation-overlap and error-correlation audit: does the learned NN
embedding capture discriminative structure that the hand-designed second-order
features (DFT/AC/RL-C9/RL-C6) do not, and are its failures complementary?

READ-ONLY: loads the real v3 checkpoint and split file, never writes to them.
Reuses the exact extraction/ranking code already validated in
eval_all_features_nn_testsplit.py / audit_all_features_dedup_testsplit.py, on
the de-duplicated 2488-probe test set (5 byte-identical duplicate-image probes
excluded, per audit_nn_v3_dedup_metrics.json / audit_all_features_dedup_testsplit.json).

Part 1 (representation overlap): classical CCA (via SVD of the whitened
cross-covariance, NOT sklearn's iterative PLS-style CCA) between the NN's
FLOAT embedding and each hand-designed feature's probe vectors, plus 5-fold
cross-validated linear-regression R^2 predicting the NN embedding from each
feature set (in-sample R^2 also reported, flagged as optimistic for the
high-dimensional feature sets).

Part 2 (error correlation): per-probe rank vectors -- NN uses its QUANTIZED
(deployment-realistic) ranking, matching every other headline number in this
project -- Spearman correlation, Jaccard overlap of failure sets at K=10/50,
and rescue counts in both directions.

Output: level1-features/results/audit_nn_interpretability.json (new file).
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ac_features import extract_ac_vector  # noqa: E402
from dft_spectrum import code_magnitude_spectrum, load_template  # noqa: E402
from rl_features import extract_rl_vector  # noqa: E402
from stage2_evaluate import ssd_matrix as ssd_matrix_np  # noqa: E402 -- (gallery, probe) signature, same as hand-features script
from nn_dataset import embed_identities, embed_all_probes, available_stems, load_template_mask, to_input_tensor  # noqa: E402
from nn_model import CircularConvEncoder  # noqa: E402
from nn_split import load_split  # noqa: E402
from nn_train import ssd_matrix as ssd_matrix_nn, ranks_of_true_identity, recall_at_k  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
_CASIA_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS_DIR = _ROOT / "level1-features" / "results"
_SPLIT_PATH = _RESULTS_DIR / "nn_identity_split.json"
_CKPT_PATH = _RESULTS_DIR / "nn_level1_v3_checkpoint_best.pt"
_OUT_PATH = _RESULTS_DIR / "audit_nn_interpretability.json"

DFT_BINS = np.arange(30, 51)
HAND_FEATURE_NAMES = ("DFT", "AC", "RL-C9", "RL-C6")
DUPLICATE_PROBES = {("029_L", "3"), ("230_R", "5"), ("236_R", "2"), ("543_L", "2"), ("703_L", "7")}
N_CV_FOLDS = 5
CCA_RIDGE = 1e-6
RNG_SEED = 2024


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(_ROOT / "level1-features"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def load_mask_2d(subject_dir: Path, stem: str) -> np.ndarray:
    with open(subject_dir / f"{stem}_mask.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def extract_features_maskaware(template: np.ndarray, mask: np.ndarray) -> dict:
    return {
        "DFT": code_magnitude_spectrum(template, row_agg="average", mask=mask)[DFT_BINS],
        "AC": extract_ac_vector(template, mask, mode="concat"),
        "RL-C9": extract_rl_vector(template, C=9, mask=mask).astype(np.float64),
        "RL-C6": extract_rl_vector(template, C=6, mask=mask).astype(np.float64),
    }


# --------------------------------------------------------------------------
# Part 1: classical CCA (canonical correlations via SVD of whitened cross-cov)
# --------------------------------------------------------------------------

def classical_cca(X: np.ndarray, Y: np.ndarray, ridge: float = CCA_RIDGE):
    """Canonical correlations between X (n,p) and Y (n,q), via eigen-whitening
    + SVD. Canonical correlations are invariant to any full-rank per-side
    linear transform, so no column standardization is needed for correctness
    (only centering + a small ridge for numerical stability on near-singular
    covariance estimates, e.g. RL-C9's 576 dims from 2488 samples)."""
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Sxx = (Xc.T @ Xc) / (n - 1) + ridge * np.eye(X.shape[1])
    Syy = (Yc.T @ Yc) / (n - 1) + ridge * np.eye(Y.shape[1])
    Sxy = (Xc.T @ Yc) / (n - 1)

    def inv_sqrt(S):
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    Sxx_inv_sqrt = inv_sqrt(Sxx)
    Syy_inv_sqrt = inv_sqrt(Syy)
    M = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
    singular_values = np.linalg.svd(M, compute_uv=False)
    correlations = np.clip(singular_values, 0.0, 1.0)
    return correlations


# --------------------------------------------------------------------------
# Part 1: 5-fold CV linear regression R^2, predicting NN embedding from a
# hand-designed feature set (average R^2 across the 64 NN output dims)
# --------------------------------------------------------------------------

def cv_and_insample_r2(X: np.ndarray, Y: np.ndarray, n_folds: int, seed: int):
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fold_sizes = np.full(n_folds, n // n_folds)
    fold_sizes[: n % n_folds] += 1
    fold_bounds = np.cumsum(fold_sizes)
    folds = np.split(perm, fold_bounds[:-1])

    def fit_predict(X_tr, Y_tr, X_te):
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
        sd_safe = np.where(sd < 1e-8, 1.0, sd)
        X_tr_std = (X_tr - mu) / sd_safe
        X_te_std = (X_te - mu) / sd_safe
        X_tr_aug = np.hstack([X_tr_std, np.ones((X_tr_std.shape[0], 1))])
        X_te_aug = np.hstack([X_te_std, np.ones((X_te_std.shape[0], 1))])
        B, *_ = np.linalg.lstsq(X_tr_aug, Y_tr, rcond=None)
        return X_te_aug @ B

    # cross-validated R^2 per output dim, averaged
    Y_pred_cv = np.zeros_like(Y)
    for te_idx in folds:
        tr_idx = np.setdiff1d(perm, te_idx, assume_unique=False)
        Y_pred_cv[te_idx] = fit_predict(X[tr_idx], Y[tr_idx], X[te_idx])
    ss_res_cv = ((Y - Y_pred_cv) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2_cv_per_dim = 1.0 - ss_res_cv / np.clip(ss_tot, 1e-12, None)

    # in-sample R^2 per output dim (fit and evaluate on all data -- optimistic, reported for reference)
    Y_pred_in = fit_predict(X, Y, X)
    ss_res_in = ((Y - Y_pred_in) ** 2).sum(axis=0)
    r2_in_per_dim = 1.0 - ss_res_in / np.clip(ss_tot, 1e-12, None)

    return float(np.mean(r2_cv_per_dim)), float(np.mean(r2_in_per_dim))


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output {_OUT_PATH}")

    device = torch.device("cpu")
    split = load_split(_SPLIT_PATH)
    train_ids, val_ids = split["identities"]["train"], split["identities"]["val"]
    test_ids = sorted(split["identities"]["test"])
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)
    assert len(test_ids) == 299 and len(full_gallery_ids) == 1998

    # ---- canonical probe ordering (identity, stem), duplicates excluded ----
    canonical_probes = []
    for ident in test_ids:
        d = _CASIA_DIR / ident
        for stem in available_stems(d):
            if stem == "1" or (ident, stem) in DUPLICATE_PROBES:
                continue
            canonical_probes.append((ident, stem))
    n_probes = len(canonical_probes)
    print(f"Canonical probe set: {n_probes} probes (expected 2488)")
    assert n_probes == 2488

    # ---- NN: float embeddings (Part 1) + quantized embeddings/ranks (Part 2) ----
    print("Loading NN checkpoint and embedding gallery + canonical probes ...")
    ckpt = torch.load(_CKPT_PATH, map_location=device, weights_only=False)
    model = CircularConvEncoder(hidden_channels=(16, 32, 64), embedding_dim=ckpt["embedding_dim"],
                                 quant_range=ckpt["quant_range"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    gallery_emb_f, gallery_ids = embed_identities(model, _CASIA_DIR, full_gallery_ids, "1", device)

    nn_probe_batch = []
    for ident, stem in canonical_probes:
        t, m = load_template_mask(_CASIA_DIR / ident, stem)
        nn_probe_batch.append(to_input_tensor(t, m))
    with torch.no_grad():
        nn_probe_emb_f = model(torch.stack(nn_probe_batch)).numpy()

    q_gallery_emb = model.quantize(torch.from_numpy(gallery_emb_f)).numpy()
    q_probe_emb = model.quantize(torch.from_numpy(nn_probe_emb_f)).numpy()

    probe_true_ids = [ident for ident, _ in canonical_probes]
    D_nn = ssd_matrix_nn(q_gallery_emb.astype(float), q_probe_emb.astype(float))
    ranks_nn = ranks_of_true_identity(D_nn, gallery_ids, probe_true_ids)
    print(f"NN quantized: R@10={recall_at_k(ranks_nn,10):.4f} median={np.median(ranks_nn)} "
          f"(sanity check vs audit_nn_v3_dedup_metrics.json: expect R@10=0.7432, median=2.0)")

    # ---- Hand-designed features: gallery + canonical probes, same order ----
    print("Extracting hand-designed features for gallery + canonical probes ...")
    gallery_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for identity in full_gallery_ids:
        d = _CASIA_DIR / identity
        feats = extract_features_maskaware(load_template(d, "1"), load_mask_2d(d, "1"))
        for name in HAND_FEATURE_NAMES:
            gallery_vecs[name].append(feats[name])
    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}

    probe_vecs = {name: [] for name in HAND_FEATURE_NAMES}
    for ident, stem in canonical_probes:
        d = _CASIA_DIR / ident
        feats = extract_features_maskaware(load_template(d, stem), load_mask_2d(d, stem))
        for name in HAND_FEATURE_NAMES:
            probe_vecs[name].append(feats[name])

    ranks_hand = {}
    for name in HAND_FEATURE_NAMES:
        gallery = np.stack(gallery_vecs[name])
        probe = np.stack(probe_vecs[name])
        probe_vecs[name] = probe  # store as array for Part 1
        D = ssd_matrix_np(gallery, probe)
        row_idx = np.arange(n_probes)
        true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
        dense_ranks = np.argsort(np.argsort(D, axis=1), axis=1)
        ranks_hand[name] = dense_ranks[row_idx, true_col] + 1
        print(f"  {name}: R@10={recall_at_k(ranks_hand[name],10):.4f} median={np.median(ranks_hand[name])}")

    # ================= Part 1: representation overlap =================
    print("\n=== Part 1: CCA + regression R^2 ===")
    part1 = {}
    for name in HAND_FEATURE_NAMES:
        X = probe_vecs[name].astype(np.float64)
        Y = nn_probe_emb_f.astype(np.float64)
        n_comp = min(X.shape[1], Y.shape[1])
        corrs = classical_cca(X, Y)
        r2_cv, r2_insample = cv_and_insample_r2(X, Y, N_CV_FOLDS, RNG_SEED)
        top5 = corrs[: min(5, len(corrs))].tolist()
        part1[name] = {
            "feature_dim": int(X.shape[1]),
            "n_canonical_components": int(n_comp),
            "cca_correlations_top5": [round(c, 4) for c in top5],
            "cca_correlations_all": [round(c, 4) for c in corrs.tolist()],
            "cca_mean_correlation": round(float(np.mean(corrs)), 4),
            "regression_r2_mean_5fold_cv": round(r2_cv, 4),
            "regression_r2_mean_insample": round(r2_insample, 4),
        }
        print(f"  {name} (dim={X.shape[1]}): top-5 CCA corr={part1[name]['cca_correlations_top5']}, "
              f"R^2 CV5={r2_cv:.4f}, R^2 in-sample={r2_insample:.4f}")

    # ================= Part 2: error correlation =================
    print("\n=== Part 2: Spearman + Jaccard + rescue counts ===")
    part2 = {}
    for name in HAND_FEATURE_NAMES:
        rf = ranks_hand[name]
        rho, pval = spearmanr(ranks_nn, rf)

        entry = {"spearman_rho": round(float(rho), 4), "spearman_pvalue": float(pval)}
        for K in (10, 50):
            fail_nn = set(np.where(ranks_nn > K)[0].tolist())
            fail_feat = set(np.where(rf > K)[0].tolist())
            union = fail_nn | fail_feat
            jaccard = len(fail_nn & fail_feat) / len(union) if union else float("nan")
            succ_nn = set(np.where(ranks_nn <= K)[0].tolist())
            succ_feat = set(np.where(rf <= K)[0].tolist())
            rescue_nn_over_feat = len(succ_nn - succ_feat)   # NN succeeds, feature fails
            rescue_feat_over_nn = len(succ_feat - succ_nn)   # feature succeeds, NN fails
            entry[f"n_failures_nn_at_{K}"] = len(fail_nn)
            entry[f"n_failures_{name}_at_{K}"] = len(fail_feat)
            entry[f"jaccard_failure_overlap_at_{K}"] = round(jaccard, 4)
            entry[f"rescue_nn_succeeds_feature_fails_at_{K}"] = rescue_nn_over_feat
            entry[f"rescue_feature_succeeds_nn_fails_at_{K}"] = rescue_feat_over_nn
        part2[name] = entry
        print(f"  {name}: Spearman rho={entry['spearman_rho']:.4f}, "
              f"Jaccard@10={entry['jaccard_failure_overlap_at_10']:.4f}, "
              f"Jaccard@50={entry['jaccard_failure_overlap_at_50']:.4f}, "
              f"rescue(NN>feat)@10={entry['rescue_nn_succeeds_feature_fails_at_10']}, "
              f"rescue(feat>NN)@10={entry['rescue_feature_succeeds_nn_fails_at_10']}")

    report = {
        "metadata": {
            "audit_date": date.today().isoformat(),
            "git_commit": _git_commit(),
            "purpose": "Representation-overlap (CCA, regression R^2) and error-correlation (Spearman, "
                       "Jaccard failure overlap, rescue counts) between the learned NN embedding and each "
                       "hand-designed Level-1 feature, on the de-duplicated 2488-probe held-out test set.",
            "n_probes": n_probes,
            "n_gallery": len(gallery_ids),
            "duplicate_probes_excluded": sorted(f"{a}_{b}" for a, b in DUPLICATE_PROBES),
            "checkpoint": str(_CKPT_PATH),
            "split_path": str(_SPLIT_PATH),
            "nn_representation_part1": "FLOAT (pre-quantization, tanh-bounded) embeddings -- representation "
                                        "structure is a property of the continuous embedding; quantization is "
                                        "a downstream deployment step and would only add discretization noise "
                                        "to a linear-structure analysis.",
            "nn_representation_part2": "QUANTIZED (deployment-realistic, +/-127) embeddings and SSD ranks -- "
                                        "matches every other headline NN number reported in this project "
                                        "(e.g. audit_nn_v3_dedup_metrics.json).",
            "cca_method": "classical CCA via SVD of the whitened cross-covariance matrix (eigen-decomposition "
                          f"based), NOT sklearn's iterative PLS-style CCA. Ridge={CCA_RIDGE} added to each "
                          "within-set covariance matrix for numerical stability. Canonical correlations are "
                          "invariant to any full-rank per-side linear transform, so no feature standardization "
                          "was needed for CCA correctness (only centering + the ridge term).",
            "regression_method": f"OLS (via numpy lstsq) predicting all 64 NN float embedding dims jointly "
                                  f"from z-scored hand-designed feature dims (z-scoring uses TRAIN-FOLD "
                                  f"statistics only, no leakage). Reports both {N_CV_FOLDS}-fold "
                                  "cross-validated R^2 (unbiased) and in-sample R^2 (fit and evaluated on all "
                                  "data -- optimistic, especially for higher-dimensional feature sets like "
                                  "RL-C9's 576 dims; reported for reference only, CV number is the honest one).",
            "hand_designed_mask_handling": "identical to eval_all_features_nn_testsplit.py / "
                                            "audit_all_features_dedup_testsplit.py (mask-aware DFT bins 30-50, "
                                            "AC concat, RL-C9/RL-C6 valid-run-only histograms).",
            "distance_metric_for_ranks": "SSD for both NN and all hand-designed features (matched, per project convention).",
        },
        "part1_representation_overlap": part1,
        "part2_error_correlation": part2,
    }

    with open(_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {_OUT_PATH}")


if __name__ == "__main__":
    main()
