#!/usr/bin/env python3
"""Loading and triplet-sampling utilities for the learned Level-1 feature network.

Loads (32, 512) binary template/mask pairs from
casia-extraction/casia-codes-2d/<identity>/{stem}_template.txt and
{stem}_mask.txt (stem '1' = gallery, '2'..'10' = up to 9 probes), using the
same "32 lines of '0'/'1' characters" loader convention as
stage2_evaluate.py / ac_features.py / dft_spectrum.py.

Each template/mask pair is converted to a 2-channel float tensor for
CircularConvEncoder (see nn_model.py):
  channel 0: bipolar code x = 2*bit - 1, zeroed at occluded positions
  channel 1: validity mask (0/1), so the network can learn which positions
             to trust rather than being told a pre-computed summary of them
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "load_template_mask",
    "to_input_tensor",
    "available_stems",
    "build_identity_index",
    "TripletIrisDataset",
    "PairBatchDataset",
    "embed_identities",
    "embed_all_probes",
]

N_ROWS, N_COLS = 32, 512


def load_template_mask(identity_dir: Path, stem: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(identity_dir / f"{stem}_template.txt") as f:
        template = np.array([list(r.strip()) for r in f], dtype=np.uint8)
    with open(identity_dir / f"{stem}_mask.txt") as f:
        mask = np.array([list(r.strip()) for r in f], dtype=np.uint8)
    return template, mask


def to_input_tensor(template: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    """(32,512) uint8 template/mask -> (2,32,512) float32 tensor, see module docstring."""
    m = mask.astype(np.float32)
    bipolar = (2.0 * template.astype(np.float32) - 1.0) * m
    x = np.stack([bipolar, m], axis=0)
    return torch.from_numpy(x)


def available_stems(identity_dir: Path) -> List[str]:
    """Sorted (numeric) list of stems with both a template and mask file present."""
    stems = []
    for p in identity_dir.glob("*_template.txt"):
        stem = p.name[: -len("_template.txt")]
        if (identity_dir / f"{stem}_mask.txt").exists():
            stems.append(stem)
    return sorted(stems, key=int)


def build_identity_index(casia_dir: Path, identities: List[str]) -> Dict[str, List[str]]:
    """identity -> sorted list of available stems, for the given identities only."""
    casia_dir = Path(casia_dir)
    return {ident: available_stems(casia_dir / ident) for ident in identities}


class TripletIrisDataset(Dataset):
    """Yields (anchor, positive, negative) input tensors for triplet-margin training.

    Anchor is always an identity's gallery image (stem '1'); positive is a
    random probe (stem != '1') of the SAME identity; negative is the gallery
    image of a different, randomly chosen identity from the same pool.

    Only identities with a gallery image AND at least one probe are usable
    as anchors (an identity with just a lone gallery image can't form a
    positive pair) -- see `usable`. Identities with no gallery image at all
    (524_L, 668_L) are expected to already be excluded by nn_split.py.

    `__len__` is a virtual epoch length (`steps_per_epoch`); each call to
    `__getitem__` draws a fresh random triplet, so the index argument itself
    is ignored beyond fixing dataset length for the DataLoader.
    """

    def __init__(self, casia_dir: Path, identities: List[str], steps_per_epoch: int, seed: int = 0):
        self.casia_dir = Path(casia_dir)
        self.index = build_identity_index(self.casia_dir, identities)
        self.usable = [ident for ident, stems in self.index.items() if "1" in stems and len(stems) >= 2]
        if len(self.usable) < 2:
            raise ValueError(
                f"need >=2 usable identities (gallery + >=1 probe) to sample triplets, got {len(self.usable)}"
            )
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int):
        anchor_id = self.rng.choice(self.usable)
        pos_stem = self.rng.choice([s for s in self.index[anchor_id] if s != "1"])
        neg_id = anchor_id
        while neg_id == anchor_id:
            neg_id = self.rng.choice(self.usable)

        a_t, a_m = load_template_mask(self.casia_dir / anchor_id, "1")
        p_t, p_m = load_template_mask(self.casia_dir / anchor_id, pos_stem)
        n_t, n_m = load_template_mask(self.casia_dir / neg_id, "1")

        return to_input_tensor(a_t, a_m), to_input_tensor(p_t, p_m), to_input_tensor(n_t, n_m)


class PairBatchDataset(Dataset):
    """Yields one full (probe_batch, gallery_batch) training batch per __getitem__ call,
    for `batch_size` DISTINCT identities sampled without replacement.

    Used with nn_model.BatchHardSSDTripletLoss: distinctness guarantees every
    other row in the batch is a genuine impostor candidate, so the hardest
    in-batch gallery embedding can be mined as the negative at zero extra
    forward-pass cost. Row i of the returned probe batch and row i of the
    gallery batch always belong to the same identity (probe = positive
    target's own gallery, mirroring the actual deployment comparison
    direction -- see BatchHardSSDTripletLoss's docstring).

    `__len__` is a virtual epoch length (`steps_per_epoch`); each call to
    `__getitem__` draws a fresh random batch of identities, so the index
    argument is ignored beyond fixing dataset length for the DataLoader. Use
    with `DataLoader(dataset, batch_size=None, ...)` since each item IS
    already a full batch.
    """

    def __init__(self, casia_dir: Path, identities: List[str], batch_size: int,
                 steps_per_epoch: int, seed: int = 0):
        self.casia_dir = Path(casia_dir)
        self.index = build_identity_index(self.casia_dir, identities)
        self.usable = [ident for ident, stems in self.index.items() if "1" in stems and len(stems) >= 2]
        if len(self.usable) < batch_size:
            raise ValueError(
                f"need >= batch_size ({batch_size}) usable identities (gallery + >=1 probe), "
                f"got {len(self.usable)}"
            )
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int):
        ids = self.rng.sample(self.usable, self.batch_size)  # without replacement -> distinct identities
        probes, galleries = [], []
        for ident in ids:
            pos_stem = self.rng.choice([s for s in self.index[ident] if s != "1"])
            p_t, p_m = load_template_mask(self.casia_dir / ident, pos_stem)
            g_t, g_m = load_template_mask(self.casia_dir / ident, "1")
            probes.append(to_input_tensor(p_t, p_m))
            galleries.append(to_input_tensor(g_t, g_m))
        return torch.stack(probes), torch.stack(galleries)


def embed_identities(model, casia_dir: Path, identities: List[str], stem: str,
                      device: torch.device, batch_size: int = 64):
    """Embed a single `stem` image for each identity that has it. Skips identities missing that stem.

    Returns (embeddings ndarray of shape (n, embedding_dim), identity_ids list of length n).
    """
    model.eval()
    casia_dir = Path(casia_dir)
    all_embs, all_ids = [], []
    batch, batch_ids = [], []

    def flush():
        nonlocal batch, batch_ids
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            e = model(x).cpu().numpy()
        all_embs.append(e)
        all_ids.extend(batch_ids)
        batch, batch_ids = [], []

    for ident in identities:
        d = casia_dir / ident
        if not ((d / f"{stem}_template.txt").exists() and (d / f"{stem}_mask.txt").exists()):
            continue
        t, m = load_template_mask(d, stem)
        batch.append(to_input_tensor(t, m))
        batch_ids.append(ident)
        if len(batch) == batch_size:
            flush()
    flush()

    if not all_embs:
        return np.zeros((0, model.embedding_dim), dtype=np.float32), []
    return np.concatenate(all_embs, axis=0), all_ids


def embed_all_probes(model, casia_dir: Path, identities: List[str],
                      device: torch.device, batch_size: int = 64):
    """Embed every available probe image (every stem != '1') for each identity.

    Returns (embeddings ndarray of shape (n, embedding_dim), true_identity_ids
    list of length n) where true_identity_ids[i] is the identity that owns
    embeddings[i] -- i.e. the gallery identity a correct ranking should recover.
    """
    model.eval()
    casia_dir = Path(casia_dir)
    all_embs, all_ids = [], []
    batch, batch_ids = [], []

    def flush():
        nonlocal batch, batch_ids
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            e = model(x).cpu().numpy()
        all_embs.append(e)
        all_ids.extend(batch_ids)
        batch, batch_ids = [], []

    for ident in identities:
        d = casia_dir / ident
        for stem in available_stems(d):
            if stem == "1":
                continue
            t, m = load_template_mask(d, stem)
            batch.append(to_input_tensor(t, m))
            batch_ids.append(ident)
            if len(batch) == batch_size:
                flush()
    flush()

    if not all_embs:
        return np.zeros((0, model.embedding_dim), dtype=np.float32), []
    return np.concatenate(all_embs, axis=0), all_ids
