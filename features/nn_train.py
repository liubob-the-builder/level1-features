#!/usr/bin/env python3
"""Train and evaluate the learned Level-1 feature network (CircularConvEncoder).

Per explicit instruction, all code for this neural-network Level-1 feature
lives in level1-features/features/ (rather than splitting the training
driver out into level1-features/experiments/, the usual convention for
run scripts in this repo -- see level1-features/README.md).

Training objective: BatchHardSSDTripletLoss (nn_model.py) directly on raw
SSD, the same operator the encrypted server-side Level-1 filter uses, so
there is no proxy-metric mismatch between training and deployment. Batches
are PairBatchDataset's (probe, gallery) pairs over B distinct identities;
for each probe the negative is the hardest (closest) other identity's
gallery embedding already present in the same batch, mined at zero extra
forward-pass cost. Superseded here vs. the first training run's
TripletIrisDataset+SSDTripletLoss (single random negative, gallery-vs-gallery
direction), which stalled once random negatives became trivially separated
-- see level1-features/README.md for the diagnosis.

Gallery-pool policy (see nn_split.py for why):
  - Training touches ONLY train-split identities (anchor/positive/negative
    triplets are all drawn from train).
  - Per-epoch validation ranks val-split probes against a DEV gallery pool
    of train+val identities' gallery images (never test), so no development
    decision (checkpoint selection, early stopping) is ever influenced by
    test data.
  - The optional --final-eval, run once against a chosen checkpoint, ranks
    test-split probes against the FULL train+val+test gallery pool (all
    1998 usable identities), so Recall@K / rank statistics are computed
    against the same N~=2000 candidate pool as the paper's own reported
    numbers -- only the query identities are restricted to test.

Distance metric throughout: SSD (sum of squared differences), matching the
paper's encrypted Level-1 comparison operator and every other feature module
in this directory.
"""

import argparse
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_dataset import PairBatchDataset, embed_identities, embed_all_probes
from nn_model import CircularConvEncoder, BatchHardSSDTripletLoss
from nn_split import build_split, save_split, load_split

_ROOT = Path(__file__).resolve().parent.parent.parent  # iris-recognition/
_DEFAULT_CASIA_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# SSD-based ranking / recall (same definitions as experiments/stage2_evaluate.py,
# duplicated locally per this repo's existing convention of each eval script
# owning its own copy rather than importing across the features/experiments split)
# ---------------------------------------------------------------------------

def ssd_matrix(gallery: np.ndarray, probe: np.ndarray) -> np.ndarray:
    """(Np, Ng) matrix of SSD(probe_i, gallery_j), via ||a-b||^2 = ||a||^2+||b||^2-2a.b."""
    g2 = (gallery ** 2).sum(axis=1)[None, :]
    p2 = (probe ** 2).sum(axis=1)[:, None]
    cross = probe @ gallery.T
    return p2 + g2 - 2.0 * cross


def ranks_of_true_identity(D: np.ndarray, gallery_ids, probe_true_ids) -> np.ndarray:
    gallery_index = {gid: i for i, gid in enumerate(gallery_ids)}
    ranks = np.empty(len(probe_true_ids), dtype=int)
    for i, true_id in enumerate(probe_true_ids):
        true_col = gallery_index[true_id]
        order = np.argsort(D[i])
        ranks[i] = int(np.where(order == true_col)[0][0]) + 1
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def evaluate(model, casia_dir: Path, gallery_identities, probe_identities, device) -> dict:
    """Embed gallery (stem '1') for `gallery_identities` and every probe for
    `probe_identities`, rank each probe's true identity, and report metrics."""
    gallery_emb, gallery_ids = embed_identities(model, casia_dir, gallery_identities, "1", device)
    probe_emb, probe_true_ids = embed_all_probes(model, casia_dir, probe_identities, device)

    # Probes whose true identity has no gallery embedding can't be scored (shouldn't
    # happen given nn_split.py's gallery filtering, but guard defensively).
    gallery_set = set(gallery_ids)
    keep = [i for i, tid in enumerate(probe_true_ids) if tid in gallery_set]
    probe_emb = probe_emb[keep]
    probe_true_ids = [probe_true_ids[i] for i in keep]

    D = ssd_matrix(gallery_emb, probe_emb)
    ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)

    return {
        "n_gallery": len(gallery_ids),
        "n_probes": len(probe_true_ids),
        "recall_at_10": recall_at_k(ranks, 10),
        "recall_at_50": recall_at_k(ranks, 50),
        "recall_at_100": recall_at_k(ranks, 100),
        "recall_at_300": recall_at_k(ranks, 300),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT / "level1-features"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(args.split_path)
    if split_path.exists():
        split = load_split(split_path)
        print(f"Loaded existing split from {split_path}")
    else:
        split = build_split(args.casia_dir, args.val_frac, args.test_frac, args.seed)
        save_split(split, split_path, args.casia_dir, args.val_frac, args.test_frac, args.seed)
        print(f"Built and saved new split to {split_path}")

    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    dev_gallery_ids = sorted(train_ids + val_ids)  # dev-time pool: train+val only, test untouched

    print(f"train={len(train_ids)} val={len(val_ids)} test={len(split['identities']['test'])} identities")

    torch.manual_seed(args.seed)
    dataset = PairBatchDataset(args.casia_dir, train_ids, batch_size=args.batch_size,
                                steps_per_epoch=args.steps_per_epoch, seed=args.seed)
    # batch_size=None: each dataset item IS already a full (probe_batch, gallery_batch) pair,
    # so the default DataLoader collation (which would try to stack items into a further batch
    # dimension) must be disabled.
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    hidden_channels = tuple(int(c) for c in args.hidden_channels.split(","))
    model = CircularConvEncoder(
        hidden_channels=hidden_channels, embedding_dim=args.embedding_dim, quant_range=args.quant_range
    ).to(device)
    loss_fn = BatchHardSSDTripletLoss(margin=args.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_recall_at_50 = -1.0
    best_epoch = -1
    ckpt_path = Path(args.out_prefix + "_checkpoint_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_dpos, epoch_dneg, n_batches = 0.0, 0.0, 0.0, 0
        for probe_batch, gallery_batch in loader:
            probe_batch, gallery_batch = probe_batch.to(device), gallery_batch.to(device)
            optimizer.zero_grad()
            # Single combined forward pass (batch = probes+galleries stacked) rather than two
            # separate calls: same total FLOPs, fewer kernel-launch/dispatch overheads, and
            # BatchNorm statistics are computed over the full combined batch (standard practice
            # for shared-weight metric-learning networks).
            bsz = probe_batch.shape[0]
            combined = torch.cat([probe_batch, gallery_batch], dim=0)
            e_combined = model(combined)
            e_probe, e_gallery = e_combined[:bsz], e_combined[bsz:]
            loss, d_pos, d_neg = loss_fn(e_probe, e_gallery)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_dpos += d_pos.item()
            epoch_dneg += d_neg.item()
            n_batches += 1

        epoch_loss /= n_batches
        epoch_dpos /= n_batches
        epoch_dneg /= n_batches

        val_metrics = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, args.casia_dir, dev_gallery_ids, val_ids, device)
            print(f"[epoch {epoch}] loss={epoch_loss:.4f} d_pos={epoch_dpos:.3f} d_neg={epoch_dneg:.3f} "
                  f"val R@10={val_metrics['recall_at_10']:.3f} R@50={val_metrics['recall_at_50']:.3f} "
                  f"medRank={val_metrics['median_rank']:.0f}/{val_metrics['n_gallery']}")
            if val_metrics["recall_at_50"] > best_recall_at_50:
                best_recall_at_50 = val_metrics["recall_at_50"]
                best_epoch = epoch
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "quant_range": args.quant_range,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }, ckpt_path)
        else:
            print(f"[epoch {epoch}] loss={epoch_loss:.4f} d_pos={epoch_dpos:.3f} d_neg={epoch_dneg:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_d_pos_mean": epoch_dpos,
            "train_d_neg_mean": epoch_dneg,
            "val_metrics": val_metrics,
        })

    elapsed = time.time() - t_start

    result = {
        "metadata": {
            "gallery_path": str(args.casia_dir),
            "split_path": str(split_path),
            "subject_count": split["metadata"]["subject_count"] if "metadata" in split else None,
            "seeds_used": {"split_seed": args.seed, "torch_seed": args.seed},
            "git_commit": _git_commit(),
            "date": date.today().isoformat(),
            "elapsed_seconds": elapsed,
        },
        "hyperparameters": {
            "embedding_dim": args.embedding_dim,
            "quant_range": args.quant_range,
            "margin": args.margin,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "steps_per_epoch": args.steps_per_epoch,
            "epochs": args.epochs,
            "eval_every": args.eval_every,
        },
        "split_counts": {
            "train_identities": len(train_ids),
            "val_identities": len(val_ids),
            "test_identities": len(split["identities"]["test"]),
            "dev_gallery_identities": len(dev_gallery_ids),
        },
        "best_epoch": best_epoch,
        "best_val_recall_at_50": best_recall_at_50,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }

    history_path = Path(args.out_prefix + "_train_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved training history to {history_path}")
    print(f"Best checkpoint (epoch {best_epoch}, val R@50={best_recall_at_50:.3f}) saved to {ckpt_path}")

    if args.final_eval:
        print("Running final held-out test evaluation (full train+val+test gallery pool)...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        full_gallery_ids = sorted(train_ids + val_ids + split["identities"]["test"])
        test_metrics = evaluate(model, args.casia_dir, full_gallery_ids, split["identities"]["test"], device)
        print(f"[FINAL TEST] R@10={test_metrics['recall_at_10']:.3f} R@50={test_metrics['recall_at_50']:.3f} "
              f"R@100={test_metrics['recall_at_100']:.3f} R@300={test_metrics['recall_at_300']:.3f} "
              f"medRank={test_metrics['median_rank']:.0f}/{test_metrics['n_gallery']} "
              f"meanRank={test_metrics['mean_rank']:.1f}")
        result["final_test_metrics"] = test_metrics
        result["final_test_metrics"]["gallery_identity_count"] = len(full_gallery_ids)
        with open(history_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Updated {history_path} with final_test_metrics")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the learned Level-1 CircularConvEncoder feature.")
    parser.add_argument("--casia-dir", type=Path, default=_DEFAULT_CASIA_DIR)
    parser.add_argument("--split-path", type=Path, default=_RESULTS_DIR / "nn_identity_split.json")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--hidden-channels", type=str, default="16,32,64",
                         help="comma-separated conv channel widths, e.g. '16,32,64'")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--quant-range", type=int, default=127)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64,
                         help="also the in-batch negative pool size for BatchHardSSDTripletLoss "
                              "(must be <= usable train identity count)")
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=1)

    parser.add_argument("--out-prefix", type=str, default=str(_RESULTS_DIR / "nn_level1"))
    parser.add_argument("--final-eval", action="store_true",
                         help="After training, evaluate the best checkpoint on the held-out test "
                              "split against the full train+val+test gallery pool.")

    args = parser.parse_args()
    train(args)
