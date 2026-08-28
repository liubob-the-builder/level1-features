#!/usr/bin/env python3
"""Adversarial audit check 5: label-shuffle leakage detector.

Trains a THROWAWAY model (never touches any real checkpoint) on the exact
same architecture, real train-split images, and BatchHardSSDTripletLoss as
nn_train.py -- EXCEPT that within every sampled batch, the correspondence
between probe identity and gallery identity is randomly permuted (a
derangement, so no identity is paired with its own gallery). This destroys
the only real training signal (which gallery image belongs to which probe
identity) while leaving every other part of the pipeline (image loading,
architecture, loss, eval code) untouched. If the pipeline had some leakage
that inflates recall independent of learning genuine iris identity, a model
"trained" on this scrambled signal would still generalize above chance
on the real test-split evaluation. If the pipeline is sound, it should
collapse to close to the random-embedding floor (~10/1998 = 0.005 for R@10).

Throwaway outputs only:
  - checkpoint: scratchpad directory (never overwrites/touches any real .pt)
  - result summary: level1-features/results/audit_shuffle_test.json (new file)

Read-only w.r.t. everything else: real split file, real checkpoints untouched.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "features"))

from nn_dataset import build_identity_index, load_template_mask, to_input_tensor, embed_identities, embed_all_probes  # noqa: E402
from nn_model import CircularConvEncoder, BatchHardSSDTripletLoss  # noqa: E402
from nn_split import load_split  # noqa: E402
from nn_train import ssd_matrix, ranks_of_true_identity, recall_at_k  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent.parent
_CASIA_DIR = _ROOT / "casia-extraction" / "casia-codes-2d"
_SPLIT_PATH = _ROOT / "level1-features" / "results" / "nn_identity_split.json"
_SCRATCH_DIR = Path("/tmp/claude-1000/-home-liubobz-iris-recognition/d23402e0-69ed-44c5-a1a7-ac9d1b699f52/scratchpad")
_THROWAWAY_CKPT = _SCRATCH_DIR / "audit_nn_shuffle_checkpoint.pt"
_OUT_PATH = _ROOT / "level1-features" / "results" / "audit_shuffle_test.json"

BATCH_SIZE = 32
STEPS_PER_EPOCH = 50
EPOCHS = 5
SEED = 777  # deliberately different from any real training seed


def derangement(n: int, rng: np.random.RandomState) -> np.ndarray:
    """Random permutation of range(n) with no fixed points (rejection sampling)."""
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


class ShuffledPairBatchDataset(Dataset):
    """Same sampling as nn_dataset.PairBatchDataset, EXCEPT the gallery row
    order within each batch is derangement-shuffled relative to the probe
    row order, so probe[i] is paired with gallery[j] of a DIFFERENT identity
    (i != j always). The loss code still treats row i of each as a matched
    pair, so training receives a scrambled/meaningless identity signal."""

    def __init__(self, casia_dir, identities, batch_size, steps_per_epoch, seed):
        self.casia_dir = Path(casia_dir)
        self.index = build_identity_index(self.casia_dir, identities)
        self.usable = [i for i, stems in self.index.items() if "1" in stems and len(stems) >= 2]
        assert len(self.usable) >= batch_size, f"need >= {batch_size} usable identities, got {len(self.usable)}"
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.py_rng = __import__("random").Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        ids = self.py_rng.sample(self.usable, self.batch_size)
        probes, galleries = [], []
        for ident in ids:
            pos_stem = self.py_rng.choice([s for s in self.index[ident] if s != "1"])
            p_t, p_m = load_template_mask(self.casia_dir / ident, pos_stem)
            g_t, g_m = load_template_mask(self.casia_dir / ident, "1")
            probes.append(to_input_tensor(p_t, p_m))
            galleries.append(to_input_tensor(g_t, g_m))
        perm = derangement(self.batch_size, self.np_rng)
        galleries = [galleries[j] for j in perm]  # scramble probe<->gallery identity correspondence
        return torch.stack(probes), torch.stack(galleries), ids, [ids[j] for j in perm]


def main():
    if _OUT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output {_OUT_PATH}")
    if _THROWAWAY_CKPT.exists():
        _THROWAWAY_CKPT.unlink()  # scratch file from a prior audit attempt, safe to clear

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split(_SPLIT_PATH)
    train_ids = split["identities"]["train"]
    val_ids = split["identities"]["val"]
    test_ids = split["identities"]["test"]
    full_gallery_ids = sorted(train_ids + val_ids + test_ids)

    torch.manual_seed(SEED)
    dataset = ShuffledPairBatchDataset(_CASIA_DIR, train_ids, BATCH_SIZE, STEPS_PER_EPOCH, SEED)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    # verify the scrambling is real before spending any training time on it
    probe_batch0, gallery_batch0, probe_ids0, gallery_ids0 = dataset[0]
    n_accidental_matches = sum(p == g for p, g in zip(probe_ids0, gallery_ids0))
    print(f"Sanity: batch-0 probe/gallery identity correspondence -- "
          f"{n_accidental_matches}/{BATCH_SIZE} accidentally still matched (should be 0)")
    assert n_accidental_matches == 0, "derangement failed -- shuffle test would be invalid"

    model = CircularConvEncoder(hidden_channels=(16, 32, 64), embedding_dim=64, quant_range=127).to(device)
    loss_fn = BatchHardSSDTripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t0 = time.time()
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        ep_loss = ep_dpos = ep_dneg = 0.0
        n_batches = 0
        for probe_batch, gallery_batch, _, _ in loader:
            probe_batch, gallery_batch = probe_batch.to(device), gallery_batch.to(device)
            optimizer.zero_grad()
            bsz = probe_batch.shape[0]
            combined = torch.cat([probe_batch, gallery_batch], dim=0)
            e_combined = model(combined)
            e_probe, e_gallery = e_combined[:bsz], e_combined[bsz:]
            loss, d_pos, d_neg = loss_fn(e_probe, e_gallery)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); ep_dpos += d_pos.item(); ep_dneg += d_neg.item(); n_batches += 1
        ep_loss /= n_batches; ep_dpos /= n_batches; ep_dneg /= n_batches
        print(f"[shuffle epoch {epoch}] loss={ep_loss:.4f} d_pos={ep_dpos:.3f} d_neg={ep_dneg:.3f}")
        history.append({"epoch": epoch, "train_loss": ep_loss, "train_d_pos_mean": ep_dpos, "train_d_neg_mean": ep_dneg})
    elapsed = time.time() - t0

    torch.save({"model_state_dict": model.state_dict(), "embedding_dim": 64, "quant_range": 127}, _THROWAWAY_CKPT)
    print(f"Saved THROWAWAY checkpoint to {_THROWAWAY_CKPT} ({elapsed:.1f}s training)")

    # Evaluate with the REAL, unmodified, correct protocol against the REAL test split.
    model.eval()
    gallery_emb, gallery_ids = embed_identities(model, _CASIA_DIR, full_gallery_ids, "1", device)
    probe_emb, probe_true_ids = embed_all_probes(model, _CASIA_DIR, test_ids, device)
    q_gallery = model.quantize(torch.from_numpy(gallery_emb)).numpy()
    q_probe = model.quantize(torch.from_numpy(probe_emb)).numpy()

    D = ssd_matrix(q_gallery, q_probe)
    ranks = ranks_of_true_identity(D, gallery_ids, probe_true_ids)
    metrics = {
        "n_gallery": len(gallery_ids), "n_probes": len(probe_true_ids),
        "recall_at_10": recall_at_k(ranks, 10), "recall_at_50": recall_at_k(ranks, 50),
        "recall_at_100": recall_at_k(ranks, 100), "recall_at_300": recall_at_k(ranks, 300),
        "median_rank": float(np.median(ranks)), "mean_rank": float(np.mean(ranks)),
    }
    print("Shuffled-label model, evaluated on REAL test split:", json.dumps(metrics, indent=2))

    chance_r10 = 10.0 / len(gallery_ids)
    chance_r50 = 50.0 / len(gallery_ids)
    report = {
        "metadata": {
            "audit_date": date.today().isoformat(),
            "purpose": "label-shuffle leakage detector: gallery<->probe identity correspondence "
                       "derangement-scrambled within every training batch, so training receives no "
                       "genuine identity signal; everything else (images, architecture, loss, eval "
                       "protocol) identical to the real pipeline.",
            "throwaway_checkpoint_path": str(_THROWAWAY_CKPT),
            "note": "This checkpoint is NOT a real result and is not saved under level1-features/results/.",
            "hyperparameters": {"batch_size": BATCH_SIZE, "steps_per_epoch": STEPS_PER_EPOCH,
                                 "epochs": EPOCHS, "seed": SEED, "elapsed_seconds": elapsed},
            "sanity_check_accidental_matches_in_batch0": int(n_accidental_matches),
        },
        "shuffle_trained_model_test_metrics": metrics,
        "chance_floor_recall_at_10": chance_r10,
        "chance_floor_recall_at_50": chance_r50,
        "real_v3_recall_at_10_for_comparison": 0.7437,
        "collapsed_to_near_chance": metrics["recall_at_10"] < 5 * chance_r10,
        "history": history,
    }
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {_OUT_PATH}")


if __name__ == "__main__":
    main()
