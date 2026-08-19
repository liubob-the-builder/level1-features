#!/usr/bin/env python3
"""Learned, rotation-invariant, mask-aware, BFV-compatible Level-1 iris feature.

Alternative to the paper's hand-crafted Level-1 features (run-length
histograms, autocorrelation lag-energy -- see ac_features.py / rl_features.py)
for Karakosta & Knottenbelt's two-level FHE-FIS filtering framework. Instead
of summarising each row with a fixed formula before learning anything, a
small CNN reads the raw (32, 512) bit code directly, and rotation-invariance
is built into the ARCHITECTURE rather than the input, so the network is not
capped at re-weighting an already-lossy hand-crafted summary:

  1. Every conv layer uses CIRCULAR padding along the 512-column angular
     axis (columns wrap, matching in-plane eye rotation = circular column
     shift) and ordinary zero padding along the 32-row axis (rows are
     radial bands / Gabor scales, not periodic). This makes the conv stack
     exactly EQUIVARIANT to circular shifts of the input: shifting the input
     by k columns shifts every feature map by exactly k columns, with no
     approximation.
  2. The final step is a MASKED, VALIDITY-WEIGHTED average over the angular
     axis, generalising the paper's own masked circular autocorrelation
     (Eq. 1: R(l) = sum_theta m(theta)m(theta+l)x(theta)x(theta+l) / sum
     m(theta)m(theta+l)) -- a masked circular average -- to a LEARNED
     per-position value instead of a fixed lag-product. A weighted average
     over a *complete* cyclic axis doesn't depend on the order of its terms,
     so this pooling step is exactly INVARIANT to circular shifts, not just
     equivariant. Equivariant conv stack -> invariant pool = the whole
     embedding is exactly rotation-invariant by construction, with no
     augmentation needed.
  3. Mask-awareness is therefore learned end-to-end (the mask is fed in as
     its own input channel, and gates the final pooling step), rather than
     imposed by a hand-picked formula -- occluded regions never contribute
     to the pooled embedding.
  4. The embedding is bounded (tanh) and can be quantized to a small integer
     range (`quantize`), matching what BFV encryption requires: a bounded,
     integer-valued vector that the server can score with a plain encrypted
     sum-of-squared-differences (SSD), the same operator the paper uses for
     Level-1 (no cosine similarity or other learned distance at inference --
     everything after the network runs in cleartext on the trusted client;
     only the resulting vector and the SSD comparison need to be
     BFV-compatible, exactly like the paper's own hand-crafted features are
     computed in cleartext and only encrypted afterward).

Distance metric: sum of squared differences (SSD) between two embeddings,
matching the paper's encrypted Level-1 comparison operator and every other
feature module in this directory.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CircularPadConv2d", "CircularConvEncoder", "ssd", "SSDTripletLoss", "BatchHardSSDTripletLoss"]


class CircularPadConv2d(nn.Module):
    """Conv2d with circular padding on the width (angular) axis and zero padding on height (row) axis.

    torch.nn.Conv2d's built-in `padding_mode` applies one mode to every
    spatial dimension, which can't mix circular (width) with zero (height).
    Padding is therefore applied manually in two passes before an ordinary
    padding=0 convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 7)):
        super().__init__()
        kh, kw = kernel_size
        assert kh % 2 == 1 and kw % 2 == 1, "kernel dims must be odd for symmetric same-size padding"
        self.pad_h = kh // 2
        self.pad_w = kw // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kh, kw), padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H=32, W=512)
        x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")   # wrap angular axis
        x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0.0)  # zero-pad row axis
        return self.conv(x)


class CircularConvEncoder(nn.Module):
    """Raw (2,32,512) code+mask -> bounded, rotation-invariant embedding of dimension `embedding_dim`.

    Input channel 0 is the bipolar code (masked to 0 at occluded positions),
    channel 1 is the validity mask -- see nn_dataset.to_input_tensor.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: Sequence[int] = (8, 16, 16),
        embedding_dim: int = 64,
        quant_range: int = 127,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in hidden_channels:
            layers.append(CircularPadConv2d(c_in, c_out, kernel_size=(3, 7)))
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, embedding_dim)
        self.embedding_dim = embedding_dim
        self.quant_range = quant_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 32, 512) -- channel 0 bipolar code, channel 1 validity mask
        mask = x[:, 1:2, :, :]                       # (B,1,32,512)
        feat = self.conv(x)                           # (B,C,32,512), shift-EQUIVARIANT along W

        # Per-row masked average over the angular axis (invariant along W).
        denom = mask.sum(dim=3, keepdim=True).clamp_min(1e-6)      # (B,1,32,1)
        row_pooled = (feat * mask).sum(dim=3, keepdim=True) / denom  # (B,C,32,1)
        row_pooled = row_pooled.squeeze(-1)                          # (B,C,32)

        # Validity-weighted average across rows (rows aren't periodic, so
        # this doesn't need to be shift-invariant -- it's a fixed reduction).
        row_weight = mask.squeeze(1).mean(dim=2)                    # (B,32)
        row_weight = row_weight / row_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (row_pooled * row_weight.unsqueeze(1)).sum(dim=2)  # (B,C)

        emb = torch.tanh(self.head(pooled))                         # (B, embedding_dim), bounded to (-1,1)
        return emb

    def quantize(self, emb: torch.Tensor) -> torch.Tensor:
        """Bounded float embedding -> bounded integer embedding for BFV-compatible SSD scoring."""
        q = torch.round(emb * self.quant_range)
        return torch.clamp(q, -self.quant_range, self.quant_range)


def ssd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sum of squared differences per row, matching the paper's encrypted Level-1 comparison operator."""
    d = a - b
    return (d * d).sum(dim=-1)


class SSDTripletLoss(nn.Module):
    """Margin triplet loss on raw SSD (not sqrt-Euclidean), the exact deployment metric.

    L = relu(SSD(anchor,positive) - SSD(anchor,negative) + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        d_pos = ssd(anchor, positive)
        d_neg = ssd(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean(), d_pos.mean(), d_neg.mean()


class BatchHardSSDTripletLoss(nn.Module):
    """Batch-hard margin triplet loss on raw SSD, mined from a batch of DISTINCT identities.

    Diagnosed need (see level1-features/README.md training notes): a single
    fully-random negative per anchor is trivially separated for the typical
    impostor pair (mean impostor SSD >> typical d_pos once training
    progresses), so most triplets give zero gradient and training stalls
    before resolving the harder near-collisions that actually drive Recall@K
    failures. Mining the hardest (closest) negative already present in the
    same batch fixes this at zero extra forward-pass cost.

    Also trains in the actual deployment comparison direction: `probe_emb[i]`
    is compared against `gallery_emb[i]` (its own identity, positive) and
    against every OTHER `gallery_emb[j]` in the batch (candidate negatives),
    matching Level-1 filtering's real operation of ranking one probe against
    many gallery entries by SSD -- rather than gallery-vs-gallery.

    Inputs must be batched over B DISTINCT identities (row i of probe_emb and
    row i of gallery_emb belong to the same identity); nn_dataset.PairBatchDataset
    guarantees this by sampling without replacement.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, probe_emb: torch.Tensor, gallery_emb: torch.Tensor):
        B = probe_emb.shape[0]
        p2 = (probe_emb ** 2).sum(dim=1, keepdim=True)          # (B,1)
        g2 = (gallery_emb ** 2).sum(dim=1, keepdim=True).T       # (1,B)
        cross = probe_emb @ gallery_emb.T                         # (B,B)
        D = p2 + g2 - 2.0 * cross                                  # D[i,j] = SSD(probe_i, gallery_j)

        d_pos = torch.diagonal(D)                                  # (B,)
        self_mask = torch.eye(B, dtype=torch.bool, device=D.device)
        d_neg, _ = D.masked_fill(self_mask, float("inf")).min(dim=1)  # hardest in-batch negative per row

        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean(), d_pos.mean(), d_neg.mean()
