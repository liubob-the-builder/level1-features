#!/usr/bin/env python3
"""Quantization check: how much Recall@K/rank quality survives going from the
raw float embedding (nn_model.CircularConvEncoder.forward, tanh-bounded reals)
to the integer embedding (CircularConvEncoder.quantize, BFV-compatible) that
would actually be encrypted and scored under BFV.

Loads a trained checkpoint, embeds the FULL held-out test-evaluation protocol
exactly as nn_train.py's --final-eval does (gallery = train+val+test, probes
= test only), computes Recall@K/median rank/mean rank once on the raw float
embeddings and once on the quantized integer embeddings, and reports both
side by side plus the recall delta at each K.
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import torch

from nn_dataset import embed_identities, embed_all_probes
from nn_model import CircularConvEncoder
from nn_split import load_split
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CASIA_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(_ROOT / "level1-features"), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _rank_metrics(gallery_emb, gallery_ids, probe_emb, probe_true_ids) -> dict:
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


def run(checkpoint_path: Path, split_path: Path, casia_dir: Path, hidden_channels: tuple) -> dict:
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CircularConvEncoder(
        hidden_channels=hidden_channels, embedding_dim=ckpt["embedding_dim"], quant_range=ckpt["quant_range"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split = load_split(split_path)
    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = split["identities"]["test"]
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)

    gallery_emb, gallery_ids = embed_identities(model, casia_dir, full_gallery_ids, "1", device)
    probe_emb, probe_true_ids = embed_all_probes(model, casia_dir, test_ids, device)

    q_gallery_emb = model.quantize(torch.from_numpy(gallery_emb)).numpy()
    q_probe_emb = model.quantize(torch.from_numpy(probe_emb)).numpy()

    float_metrics = _rank_metrics(gallery_emb, gallery_ids, probe_emb, probe_true_ids)
    quantized_metrics = _rank_metrics(q_gallery_emb, gallery_ids, q_probe_emb, probe_true_ids)

    delta = {
        k: quantized_metrics[k] - float_metrics[k]
        for k in ("recall_at_10", "recall_at_50", "recall_at_100", "recall_at_300")
    }
    delta["median_rank"] = quantized_metrics["median_rank"] - float_metrics["median_rank"]

    return {
        "metadata": {
            "gallery_path": str(casia_dir),
            "checkpoint_path": str(checkpoint_path),
            "split_path": str(split_path),
            "quant_range": ckpt["quant_range"],
            "embedding_dim": ckpt["embedding_dim"],
            "checkpoint_epoch": ckpt.get("epoch"),
            "git_commit": _git_commit(),
            "date": date.today().isoformat(),
        },
        "float_metrics": float_metrics,
        "quantized_metrics": quantized_metrics,
        "quantized_minus_float_delta": delta,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare float vs. BFV-compatible quantized embedding recall.")
    parser.add_argument("--checkpoint", type=Path, default=_RESULTS_DIR / "nn_level1_v2_checkpoint_best.pt")
    parser.add_argument("--split-path", type=Path, default=_RESULTS_DIR / "nn_identity_split.json")
    parser.add_argument("--casia-dir", type=Path, default=_DEFAULT_CASIA_DIR)
    parser.add_argument("--hidden-channels", type=str, default="16,32,64")
    parser.add_argument("--out", type=Path, default=_RESULTS_DIR / "nn_level1_v2_quantized_eval.json")
    args = parser.parse_args()

    hidden_channels = tuple(int(c) for c in args.hidden_channels.split(","))
    result = run(args.checkpoint, args.split_path, args.casia_dir, hidden_channels)

    print("float:     ", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["float_metrics"].items()})
    print("quantized: ", {k: round(v, 4) if isinstance(v, float) else v for k, v in result["quantized_metrics"].items()})
    print("delta (quantized - float):", {k: round(v, 4) for k, v in result["quantized_minus_float_delta"].items()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.out}")
