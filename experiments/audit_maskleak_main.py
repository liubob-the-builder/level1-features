#!/usr/bin/env python3
"""Mask-geometry leakage audit for the learned Level-1 NN embedding (v3).

Question: does the trained CircularConvEncoder embedding partly exploit
occlusion-mask GEOMETRY (eyelid/eyelash shape, which is anatomically
consistent per subject within this single-session CASIA-Iris-Thousand
dataset) rather than iris texture? Identity-disjoint train/val/test splits
alone do NOT rule this out, since mask shape is subject-correlated even
across disjoint identities' own repeated captures.

Read-only w.r.t. every existing file: loads (never modifies)
nn_level1_v3_checkpoint_best.pt and nn_identity_split.json, and writes only
to new audit_maskleak_-prefixed files. Uses the EXACT same evaluation split,
matching protocol (299 test identities / 1998 gallery / 2488 de-duplicated
probes -- see DUPLICATE_PROBES, matching audit_all_features_dedup_testsplit.py)
and metrics (Recall@10/50/100/300, median/mean rank -- no EER, this is an
identification benchmark) as all_features_nn_testsplit_comparison.json /
audit_nn_v3_dedup_metrics.json, so every number here is directly comparable.

Three checks:
  1. Mask-only baseline: rank probes using ONLY a 32-dim per-row mask
     validity-fraction vector (no model, no learning). Bounds how much
     identity information the mask carries by itself.
  2. Mask-channel swap at inference: reuses the trained model's `conv` and
     `head` submodules unmodified, but feeds a DIFFERENT, randomly chosen
     identity's mask into channel 1 (the conv stack's "soft" pathway) while
     keeping channel 0 (the iris code, computed from the TRUE mask) and the
     hard pooling-stage mask gate both TRUE. Isolates the soft-masking
     pathway from the hard pooling gate. 3 seeds, mean/std reported.
  3. Embedding-distance vs mask-distance correlation, impostor pairs only:
     Pearson/Spearman between (a) SSD over 64-dim quantized embeddings (the
     benchmark's own distance) and (b) Euclidean distance over 32-dim
     row-validity mask profiles.

Every check is preceded by a self-test:
  - selftest_ssd_ranking: ssd_matrix/ranks_of_true_identity (written for
    64-dim embeddings) are checked on an arbitrary-dimension (dim=2) toy
    example with known correct ranks, before trusting them on 32-dim mask
    vectors.
  - selftest_forward_equivalence: forward_with_mask_swap (the hand-written
    pooling reimplementation used for Check 2) must reproduce model(x)
    exactly (max abs diff < 1e-6) when swap_mask == true_mask, before any
    swap run is trusted.
  - Check 1 plausibility gate: if Recall@10 is <= 0 or >= 0.6 ("near the
    learned embedding's own 0.743"), the script stops BEFORE writing any
    final output and instead writes a *_FLAGGED.json describing why --
    per instruction, an implausible Check-1 number means "stop and tell me",
    not "write it up".
"""

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_FEATURES = _LEVEL1 / "features"
_CASIA_2D_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS = _LEVEL1 / "results"
_SPLIT_PATH = _RESULTS / "nn_identity_split.json"
_CKPT_PATH = _RESULTS / "nn_level1_v3_checkpoint_best.pt"
_DEDUP_REF_PATH = _RESULTS / "audit_nn_v3_dedup_metrics.json"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

from nn_dataset import load_template_mask, to_input_tensor, available_stems  # noqa: E402
from nn_model import CircularConvEncoder  # noqa: E402
from nn_split import load_split  # noqa: E402
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k  # noqa: E402

_OUT_JSON = _RESULTS / "audit_maskleak_results.json"
_OUT_MD = _RESULTS / "audit_maskleak_RESULTS.md"
_OUT_SCATTER = _RESULTS / "audit_maskleak_check3_scatter.png"
_OUT_FLAGGED = _RESULTS / "audit_maskleak_check1_FLAGGED.json"

K_VALUES = [10, 50, 100, 300]
DUPLICATE_PROBES = {("029_L", "3"), ("230_R", "5"), ("236_R", "2"), ("543_L", "2"), ("703_L", "7")}
EXPECTED_N_TEST_IDENTITIES = 299
EXPECTED_N_GALLERY = 1998
EXPECTED_N_PROBES = 2488
BATCH_SIZE = 64
SWAP_SEEDS = [0, 1, 2]
SCATTER_SUBSAMPLE = 20000
SCATTER_SEED = 12345
CHECK1_RECALL10_FLOOR = 1e-9      # "exactly zero" guard
CHECK1_RECALL10_CEILING = 0.6     # "near the learned embedding's" 0.743 guard


def get_git_commit(repo_dir: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def selftest_ssd_ranking():
    """ssd_matrix/ranks_of_true_identity were written for 64-dim embeddings;
    check they behave correctly on an arbitrary dimension (here dim=2) with
    hand-computable correct ranks, before trusting them on 32-dim mask
    vectors in Check 1."""
    gallery = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    gallery_ids = ["A", "B", "C"]
    probe = np.array([[0.1, 0.1], [1.9, 1.9]])
    probe_true_ids = ["A", "C"]
    D = ssd_matrix(gallery, probe)
    assert D.shape == (2, 3), f"unexpected D shape {D.shape}"
    ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)
    assert ranks.shape == (2,), f"unexpected ranks shape {ranks.shape}"
    assert list(ranks) == [1, 1], f"toy self-test failed, got ranks={list(ranks)}"
    print("[selftest] ssd_matrix/ranks_of_true_identity toy check PASSED (ranks == [1, 1])")


def forward_with_mask_swap(model, code_batch, swap_mask_batch, true_mask_batch):
    """Reimplements CircularConvEncoder.forward (nn_model.py) but decouples the
    mask fed into the conv stack (channel 1, the "soft" pathway) from the mask
    used for the hard pooling gate. Reuses the model's trained `conv` and
    `head` submodules verbatim -- nn_model.py itself is never modified; only
    the pooling wiring is hand-written here. With swap_mask_batch ==
    true_mask_batch this must reproduce model(x) exactly -- see
    selftest_forward_equivalence."""
    x = torch.stack([code_batch, swap_mask_batch], dim=1)  # (B,2,32,512)
    feat = model.conv(x)  # (B,C,32,512), shift-equivariant conv stack

    mask = true_mask_batch.unsqueeze(1)  # (B,1,32,512) -- TRUE mask, pooling gate
    denom = mask.sum(dim=3, keepdim=True).clamp_min(1e-6)
    row_pooled = (feat * mask).sum(dim=3, keepdim=True) / denom
    row_pooled = row_pooled.squeeze(-1)

    row_weight = mask.squeeze(1).mean(dim=2)
    row_weight = row_weight / row_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    pooled = (row_pooled * row_weight.unsqueeze(1)).sum(dim=2)

    emb = torch.tanh(model.head(pooled))
    return emb


def selftest_forward_equivalence(model, sample_inputs) -> float:
    """With swap_mask == true_mask, forward_with_mask_swap must match model(x)."""
    codes = torch.stack([x[0] for x in sample_inputs])
    masks = torch.stack([x[1] for x in sample_inputs])
    x_full = torch.stack(sample_inputs)
    with torch.no_grad():
        ref = model(x_full)
        test = forward_with_mask_swap(model, codes, masks, masks)
    max_diff = (ref - test).abs().max().item()
    print(f"[selftest] forward_with_mask_swap no-swap equivalence: max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-6, f"forward_with_mask_swap does not match model(x) closely enough: {max_diff}"
    print("[selftest] forward_with_mask_swap equivalence check PASSED (< 1e-6)")
    return max_diff


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_common():
    split = load_split(_SPLIT_PATH)
    train_ids, val_ids = split["identities"]["train"], split["identities"]["val"]
    test_ids = sorted(split["identities"]["test"])
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)
    assert len(test_ids) == EXPECTED_N_TEST_IDENTITIES
    assert len(full_gallery_ids) == EXPECTED_N_GALLERY

    gallery_items = []  # (identity, template, mask)
    for identity in full_gallery_ids:
        d = _CASIA_2D_DIR / identity
        t, m = load_template_mask(d, "1")
        gallery_items.append((identity, t, m))

    probe_items = []  # (identity, stem, template, mask)
    n_skipped = 0
    for identity in test_ids:
        d = _CASIA_2D_DIR / identity
        for stem in available_stems(d):
            if stem == "1":
                continue
            if (identity, stem) in DUPLICATE_PROBES:
                n_skipped += 1
                continue
            t, m = load_template_mask(d, stem)
            probe_items.append((identity, stem, t, m))
    assert n_skipped == 5, f"expected to skip 5 duplicate probes, skipped {n_skipped}"
    assert len(probe_items) == EXPECTED_N_PROBES, f"expected {EXPECTED_N_PROBES} probes, got {len(probe_items)}"

    ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
    model = CircularConvEncoder(
        hidden_channels=(16, 32, 64), embedding_dim=ckpt["embedding_dim"], quant_range=ckpt["quant_range"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return test_ids, full_gallery_ids, gallery_items, probe_items, model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_baseline(model, items, is_probe: bool, batch_size=BATCH_SIZE):
    embs, ids, batch = [], [], []

    def flush():
        nonlocal batch
        if not batch:
            return
        with torch.no_grad():
            embs.append(model(torch.stack(batch)).numpy())
        batch = []

    for item in items:
        ident, t, m = (item[0], item[2], item[3]) if is_probe else item
        batch.append(to_input_tensor(t, m))
        ids.append(ident)
        if len(batch) == batch_size:
            flush()
    flush()
    return np.concatenate(embs, axis=0), ids


def embed_with_swap(model, items, is_probe: bool, masks_by_identity, rng, batch_size=BATCH_SIZE):
    """Per image: draw a random mask from a DIFFERENT identity (uniformly,
    via rng) and feed it as channel 1 to the conv stack; channel 0 (iris
    code, from the TRUE mask) and the pooling-stage mask both stay TRUE."""
    other_identity_pool = list(masks_by_identity.keys())
    embs, ids = [], []
    code_batch, swapmask_batch, truemask_batch = [], [], []

    def flush():
        nonlocal code_batch, swapmask_batch, truemask_batch
        if not code_batch:
            return
        codes = torch.stack(code_batch)
        swaps = torch.stack(swapmask_batch)
        trues = torch.stack(truemask_batch)
        with torch.no_grad():
            e = forward_with_mask_swap(model, codes, swaps, trues).numpy()
        embs.append(e)
        code_batch, swapmask_batch, truemask_batch = [], [], []

    for item in items:
        ident, t, m = (item[0], item[2], item[3]) if is_probe else item
        x = to_input_tensor(t, m)  # channel0=bipolar(true mask), channel1=true mask
        code, true_mask = x[0], x[1]

        other_ident = ident
        while other_ident == ident:
            other_ident = other_identity_pool[rng.integers(len(other_identity_pool))]
        candidates = masks_by_identity[other_ident]
        swap_mask = candidates[rng.integers(len(candidates))]

        code_batch.append(code)
        swapmask_batch.append(torch.from_numpy(swap_mask.astype(np.float32)))
        truemask_batch.append(true_mask)
        ids.append(ident)
        if len(code_batch) == batch_size:
            flush()
    flush()
    return np.concatenate(embs, axis=0), ids


def rank_metrics(gallery_emb, gallery_ids, probe_emb, probe_true_ids):
    D = ssd_matrix(gallery_emb, probe_emb)
    ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)
    out = {
        "n_gallery": len(gallery_ids),
        "n_probes": len(probe_true_ids),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    for k in K_VALUES:
        out[f"recall_at_{k}"] = recall_at_k(ranks, k)
    return out, D


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_row(name, m):
    return (f"| {name} | {m['recall_at_10']:.4f} | {m['recall_at_50']:.4f} | "
            f"{m['recall_at_100']:.4f} | {m['recall_at_300']:.4f} | {m['median_rank']:.1f} |")


def build_markdown(baseline_metrics, check1_metrics, ref, check2_per_seed, check2_mean, check2_std,
                    n_pairs, pearson_r, pearson_p, spearman_r, spearman_p):
    lines = []
    lines.append("# Mask-geometry leakage audit -- CircularConvEncoder v3")
    lines.append("")
    lines.append("Read-only audit. Same 299-test-identity / 1998-gallery / 2488-de-duplicated-probe "
                 "protocol as `all_features_nn_testsplit_comparison.json` / `audit_nn_v3_dedup_metrics.json`. "
                 "Identification metrics only (Recall@K, median rank) -- no EER.")
    lines.append("")
    lines.append("## Reference baseline (unmodified NN v3, quantized)")
    lines.append("")
    lines.append("| | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(_fmt_row("NN v3 (unmodified)", baseline_metrics))
    lines.append("")

    lines.append("## Check 1 -- mask-only baseline (no learning)")
    lines.append("")
    lines.append("32-dim per-row mask validity fraction, ranked by SSD (same distance/harness as every "
                 "other feature). Bounds how much identity information the mask alone carries.")
    lines.append("")
    lines.append("| | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(_fmt_row("Mask-only (32-dim)", check1_metrics))
    lines.append(_fmt_row("NN v3 (unmodified)", baseline_metrics))
    lines.append("")
    lines.append(f"Random-embedding floor for reference: R@10 ~ {10 / EXPECTED_N_GALLERY:.4f}.")
    lines.append("")
    non_trivial = check1_metrics["recall_at_10"] > 0.05
    lines.append(
        f"**Implication:** the mask-only feature reaches R@10={check1_metrics['recall_at_10']:.4f} "
        f"against a ~{10 / EXPECTED_N_GALLERY:.4f} random floor and the NN's "
        f"{baseline_metrics['recall_at_10']:.4f} -- mask geometry alone carries "
        f"{'non-trivial' if non_trivial else 'very little'} identity information on its own, "
        "well short of the learned embedding's performance."
    )
    lines.append("")

    lines.append("## Check 2 -- mask-channel swap at inference (3 seeds)")
    lines.append("")
    lines.append("Channel 1 (fed to the conv stack) replaced with a different, randomly chosen "
                 "identity's mask; channel 0 (iris code) and the pooling-stage mask stay TRUE. "
                 "Isolates the soft-masking (conv) pathway from the hard pooling gate.")
    lines.append("")
    lines.append("| Seed | R@10 | R@50 | R@100 | R@300 | MedRank |")
    lines.append("|---|---|---|---|---|---|")
    for m in check2_per_seed:
        lines.append(_fmt_row(f"swap seed={m['seed']}", m))
    lines.append(f"| **mean** | {check2_mean['recall_at_10']:.4f} | {check2_mean['recall_at_50']:.4f} | "
                 f"{check2_mean['recall_at_100']:.4f} | {check2_mean['recall_at_300']:.4f} | "
                 f"{check2_mean['median_rank']:.1f} |")
    lines.append(f"| **std**  | {check2_std['recall_at_10']:.4f} | {check2_std['recall_at_50']:.4f} | "
                 f"{check2_std['recall_at_100']:.4f} | {check2_std['recall_at_300']:.4f} | "
                 f"{check2_std['median_rank']:.1f} |")
    lines.append(_fmt_row("baseline (no swap)", baseline_metrics))
    lines.append("")
    drop10 = baseline_metrics["recall_at_10"] - check2_mean["recall_at_10"]
    substantial = drop10 > 0.05
    lines.append(
        f"**Implication:** swapping channel 1 for an uncorrelated identity's mask moves R@10 from "
        f"{baseline_metrics['recall_at_10']:.4f} (baseline) to {check2_mean['recall_at_10']:.4f} "
        f"(mean over {len(check2_per_seed)} seeds, std {check2_std['recall_at_10']:.4f}) -- "
        f"{'a substantial drop, indicating real reliance on the soft (conv-stack) masking pathway' if substantial else 'little to no drop, indicating the soft masking pathway is not a major driver of recall'}."
    )
    lines.append("")

    lines.append("## Check 3 -- embedding distance vs mask distance (impostor pairs only)")
    lines.append("")
    lines.append(f"n = {n_pairs} impostor probe-gallery pairs (all pairs excluding the genuine match). "
                 "(a) SSD between 64-dim quantized embeddings (the benchmark distance); (b) Euclidean "
                 "distance between 32-dim row-validity mask profiles.")
    lines.append("")
    lines.append(f"- Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    lines.append(f"- Spearman rho = {spearman_r:.4f} (p = {spearman_p:.2e})")
    lines.append("")
    lines.append("![scatter](audit_maskleak_check3_scatter.png)")
    lines.append("")
    correlated = abs(spearman_r) > 0.3
    lines.append(
        f"**Implication:** embedding impostor-pair distances and mask-profile impostor-pair distances are "
        f"{'meaningfully correlated' if correlated else 'weakly or not correlated'} "
        f"(Spearman rho={spearman_r:.4f}) -- "
        f"{'consistent with shared mask-driven signal in the impostor-distance structure' if correlated else 'the embedding impostor-distance structure appears largely independent of mask-profile distance'}."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if _OUT_JSON.exists():
        raise FileExistsError(f"{_OUT_JSON} already exists; write a new filename instead.")

    print("=== self-tests ===", flush=True)
    selftest_ssd_ranking()

    print("\n=== loading split, checkpoint, raw templates/masks ===", flush=True)
    test_ids, full_gallery_ids, gallery_items, probe_items, model = load_common()
    print(f"gallery={len(gallery_items)} probes={len(probe_items)} (dedup, matching prior audits)", flush=True)

    sample_inputs = [to_input_tensor(t, m) for (_, t, m) in gallery_items[:8]]
    equivalence_max_diff = selftest_forward_equivalence(model, sample_inputs)

    print("\n=== baseline (unmodified) embeddings ===", flush=True)
    t0 = time.time()
    gallery_emb_f, gallery_ids = embed_baseline(model, gallery_items, is_probe=False)
    probe_emb_f, probe_true_ids = embed_baseline(model, probe_items, is_probe=True)
    gallery_emb_q = model.quantize(torch.from_numpy(gallery_emb_f)).numpy()
    probe_emb_q = model.quantize(torch.from_numpy(probe_emb_f)).numpy()
    baseline_metrics, D_baseline_q = rank_metrics(gallery_emb_q, gallery_ids, probe_emb_q, probe_true_ids)
    print(f"baseline (quantized) metrics: {baseline_metrics}  ({time.time() - t0:.1f}s)", flush=True)

    ref = json.loads(_DEDUP_REF_PATH.read_text())["quantized_deduplicated"]
    mismatches = {
        k: (baseline_metrics[k], ref[k])
        for k in ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300", "median_rank", "mean_rank")
        if not np.isclose(baseline_metrics[k], ref[k], atol=1e-9)
    }
    if mismatches:
        raise RuntimeError(f"Recomputed baseline does not match audit_nn_v3_dedup_metrics.json: {mismatches}")
    print("[sanity] recomputed baseline matches audit_nn_v3_dedup_metrics.json exactly", flush=True)

    # ---------------- CHECK 1: mask-only baseline ----------------
    print("\n=== CHECK 1: mask-only baseline ===", flush=True)

    def row_validity(mask):
        return mask.astype(np.float64).mean(axis=1)  # (32,)

    gallery_mask_vec = np.stack([row_validity(m) for (_, t, m) in gallery_items])
    probe_mask_vec = np.stack([row_validity(m) for (_, s, t, m) in probe_items])
    check1_metrics, D_mask = rank_metrics(gallery_mask_vec, gallery_ids, probe_mask_vec, probe_true_ids)
    print(f"check1 (mask-only) metrics: {check1_metrics}", flush=True)

    r10 = check1_metrics["recall_at_10"]
    if r10 <= CHECK1_RECALL10_FLOOR or r10 >= CHECK1_RECALL10_CEILING:
        flagged = {
            "reason": "Check 1 Recall@10 outside the plausible range "
                      f"({CHECK1_RECALL10_FLOOR}, {CHECK1_RECALL10_CEILING}); likely a harness bug "
                      "rather than a real result. Stopping before writing final outputs, per instruction.",
            "check1_metrics": check1_metrics,
            "random_floor_recall_at_10_reference": 10 / EXPECTED_N_GALLERY,
            "nn_recall_at_10_reference": ref["recall_at_10"],
        }
        with open(_OUT_FLAGGED, "w") as f:
            json.dump(flagged, f, indent=2)
        print(f"\n!!! CHECK 1 FLAGGED AS IMPLAUSIBLE -- see {_OUT_FLAGGED}. Stopping. !!!", flush=True)
        return
    print(f"[sanity] check1 recall@10={r10:.4f} is between the random floor "
          f"(~{10 / EXPECTED_N_GALLERY:.4f}) and the NN reference ({ref['recall_at_10']:.4f}) -- plausible.",
          flush=True)

    # ---------------- CHECK 2: mask-channel swap ----------------
    print("\n=== CHECK 2: mask-channel swap at inference ===", flush=True)
    masks_by_identity = {}
    for (ident, t, m) in gallery_items:
        masks_by_identity.setdefault(ident, []).append(m)
    for (ident, s, t, m) in probe_items:
        masks_by_identity.setdefault(ident, []).append(m)

    check2_per_seed = []
    for seed in SWAP_SEEDS:
        rng = np.random.default_rng(seed)
        t0 = time.time()
        g_emb_f, g_ids = embed_with_swap(model, gallery_items, False, masks_by_identity, rng)
        p_emb_f, p_true_ids = embed_with_swap(model, probe_items, True, masks_by_identity, rng)
        g_emb_q = model.quantize(torch.from_numpy(g_emb_f)).numpy()
        p_emb_q = model.quantize(torch.from_numpy(p_emb_f)).numpy()
        metrics, _ = rank_metrics(g_emb_q, g_ids, p_emb_q, p_true_ids)
        metrics["seed"] = seed
        check2_per_seed.append(metrics)
        print(f"  seed={seed}: {metrics}  ({time.time() - t0:.1f}s)", flush=True)

    metric_keys = ["median_rank", "mean_rank"] + [f"recall_at_{k}" for k in K_VALUES]
    check2_mean = {key: float(np.mean([m[key] for m in check2_per_seed])) for key in metric_keys}
    check2_std = {key: float(np.std([m[key] for m in check2_per_seed])) for key in metric_keys}
    print(f"check2 mean across seeds: {check2_mean}", flush=True)
    print(f"check2 std across seeds: {check2_std}", flush=True)

    # ---------------- CHECK 3: embedding distance vs mask distance ----------------
    print("\n=== CHECK 3: embedding SSD vs mask-distance correlation (impostor pairs) ===", flush=True)
    gallery_idx = {gid: i for i, gid in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[pid] for pid in probe_true_ids])
    row_idx = np.arange(len(probe_true_ids))
    impostor_mask = np.ones_like(D_baseline_q, dtype=bool)
    impostor_mask[row_idx, true_col] = False

    emb_dist = D_baseline_q[impostor_mask]
    mask_dist = np.sqrt(np.clip(D_mask[impostor_mask], 0, None))
    n_pairs = emb_dist.shape[0]
    print(f"n impostor pairs = {n_pairs}", flush=True)

    t0 = time.time()
    pearson_r, pearson_p = pearsonr(emb_dist, mask_dist)
    spearman_r, spearman_p = spearmanr(emb_dist, mask_dist)
    print(f"pearson r={pearson_r:.4f} (p={pearson_p:.2e}), spearman rho={spearman_r:.4f} (p={spearman_p:.2e}) "
          f"({time.time() - t0:.1f}s)", flush=True)

    rng3 = np.random.default_rng(SCATTER_SEED)
    n_sub = min(SCATTER_SUBSAMPLE, n_pairs)
    sub_idx = rng3.choice(n_pairs, size=n_sub, replace=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(mask_dist[sub_idx], emb_dist[sub_idx], s=2, alpha=0.25)
    ax.set_xlabel("Euclidean distance, 32-dim row-validity mask profile")
    ax.set_ylabel("SSD distance, 64-dim quantized embedding")
    ax.set_title(f"Impostor pairs: embedding vs mask distance (n={n_sub} of {n_pairs} shown)\n"
                 f"Pearson r={pearson_r:.3f}, Spearman rho={spearman_r:.3f} (computed on full set)")
    fig.tight_layout()
    fig.savefig(_OUT_SCATTER, dpi=150)
    plt.close(fig)
    print(f"saved scatter to {_OUT_SCATTER}", flush=True)

    # ---------------- write consolidated JSON ----------------
    result = {
        "metadata": {
            "purpose": "Mask-geometry leakage audit for CircularConvEncoder v3: tests whether the "
                       "trained embedding partly exploits occlusion-mask geometry rather than iris "
                       "texture, since mask shape is subject-correlated within this single-session "
                       "dataset and identity-disjoint splits alone don't rule that out.",
            "checkpoint_path": str(_CKPT_PATH.relative_to(_ROOT)),
            "split_path": str(_SPLIT_PATH.relative_to(_ROOT)),
            "reference_dedup_metrics": str(_DEDUP_REF_PATH.relative_to(_ROOT)),
            "n_gallery": EXPECTED_N_GALLERY,
            "n_probes": EXPECTED_N_PROBES,
            "swap_seeds": SWAP_SEEDS,
            "swap_pool": "each image's channel-1 mask is drawn uniformly, per image, from any OTHER "
                         "identity's available image(s) (gallery+probe pool combined)",
            "scatter_subsample": n_sub,
            "scatter_seed": SCATTER_SEED,
            "distance_metric": "SSD (embeddings), Euclidean (32-dim row-validity mask profile)",
            "git_commit": get_git_commit(_LEVEL1),
            "date": date.today().isoformat(),
        },
        "selftests": {
            "ssd_ranking_toy_check": "PASSED (ranks == [1, 1] on a hand-verified 2-dim toy example)",
            "forward_with_mask_swap_equivalence_max_abs_diff": equivalence_max_diff,
        },
        "baseline_quantized_metrics": baseline_metrics,
        "baseline_matches_dedup_reference": True,
        "check1_mask_only_baseline": {
            **check1_metrics,
            "random_floor_recall_at_10_reference": 10 / EXPECTED_N_GALLERY,
            "nn_recall_at_10_reference": ref["recall_at_10"],
        },
        "check2_mask_channel_swap": {
            "per_seed": check2_per_seed,
            "mean": check2_mean,
            "std": check2_std,
            "baseline_for_comparison": baseline_metrics,
        },
        "check3_distance_correlation": {
            "n_impostor_pairs": int(n_pairs),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_r),
            "spearman_p": float(spearman_p),
            "scatter_path": str(_OUT_SCATTER.relative_to(_ROOT)),
        },
    }
    with open(_OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {_OUT_JSON}", flush=True)

    md = build_markdown(baseline_metrics, check1_metrics, ref, check2_per_seed, check2_mean, check2_std,
                         n_pairs, pearson_r, pearson_p, spearman_r, spearman_p)
    with open(_OUT_MD, "w") as f:
        f.write(md)
    print(f"Saved {_OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
