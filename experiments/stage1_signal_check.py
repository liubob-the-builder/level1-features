#!/usr/bin/env python3
"""Stage 1 (pre-Stage-2 gate) — quick genuine-vs-impostor separation check.

Builds the DFT-magnitude Level-1 feature (per-row FFT along the 512-column
angular axis, magnitude, averaged across 32 rows, DC dropped) restricted to
two candidate bin sets:
  A: bins 30-50
  B: bins 30-50 plus 78-82

For each of the 300 subjects in sicgen_gallery_300subjects/, computes the
genuine SSD (IC1 vs IC2, same subject) and the impostor SSD (IC1_i vs IC2_j,
i != j). Impostor pairs are computed exhaustively (all 300*299 = 89,700
off-diagonal pairs) rather than a small random sample, since it's cheap at
this feature dimensionality and gives a more precise estimate than sampling
a few hundred.

Reports mean/std of both distributions and d' per bin set, and plots
overlaid histograms. This does NOT run ranking/recall -- it's only a
signal-presence gate before Stage 2.
"""

import sys
from pathlib import Path
# Anchor to repo layout regardless of working directory:
# this file is in level1-features/experiments/ ; galleries are in ../../sic-gen/,
# feature modules are in ../features/
_SICGEN = Path(__file__).resolve().parent.parent.parent / "sic-gen"
_FEATURES = Path(__file__).resolve().parent.parent / "features"
sys.path.insert(0, str(_FEATURES))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dft_spectrum import batch_code_spectra

GALLERY_DIR = _SICGEN / "sicgen_gallery_300subjects"
OUT_PATH = Path("stage1_signal_check.png")

BIN_SETS = {
    "A: bins 30-50": np.arange(30, 51),
    "B: bins 30-50 + 78-82": np.concatenate([np.arange(30, 51), np.arange(78, 83)]),
}


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return abs(impostor.mean() - genuine.mean()) / np.sqrt(0.5 * (genuine.var() + impostor.var()))


def main():
    spectra1, dirs1 = batch_code_spectra(GALLERY_DIR, stem="1", row_agg="average")
    spectra2, dirs2 = batch_code_spectra(GALLERY_DIR, stem="2", row_agg="average")
    assert [d.name for d in dirs1] == [d.name for d in dirs2], "IC1/IC2 subject ordering mismatch"
    n = len(dirs1)
    print(f"Loaded {n} subjects (IC1 gallery, IC2 probe).")

    fig, axes = plt.subplots(1, len(BIN_SETS), figsize=(7 * len(BIN_SETS), 5), sharey=True)
    if len(BIN_SETS) == 1:
        axes = [axes]

    for ax, (label, bins) in zip(axes, BIN_SETS.items()):
        feat1 = spectra1[:, bins]  # (n, len(bins)) -- gallery (IC1)
        feat2 = spectra2[:, bins]  # (n, len(bins)) -- probe   (IC2)

        # Full SSD matrix: rows = gallery subject i, cols = probe subject j
        diff = feat1[:, None, :] - feat2[None, :, :]
        ssd_matrix = np.sum(diff ** 2, axis=-1)  # (n, n)

        genuine = np.diag(ssd_matrix)
        impostor = ssd_matrix[~np.eye(n, dtype=bool)]

        dp = d_prime(genuine, impostor)
        print(f"\n{label}  (dim={len(bins)})")
        print(f"  Genuine  SSD: mean={genuine.mean():.3f}  std={genuine.std():.3f}  n={genuine.size}")
        print(f"  Impostor SSD: mean={impostor.mean():.3f}  std={impostor.std():.3f}  n={impostor.size}")
        print(f"  d' = {dp:.3f}")

        bins_hist = np.linspace(0, np.percentile(impostor, 99.5), 60)
        ax.hist(genuine, bins=bins_hist, density=True, alpha=0.75,
                color="white", edgecolor="green", linewidth=1.5,
                label=f"Genuine  μ={genuine.mean():.1f} σ={genuine.std():.1f}")
        ax.hist(impostor, bins=bins_hist, density=True, alpha=0.75,
                color="white", edgecolor="red", linewidth=1.5,
                label=f"Impostor μ={impostor.mean():.1f} σ={impostor.std():.1f}")
        ax.set_xlabel("SSD", size=12)
        ax.set_title(f"{label}\nd' = {dp:.3f}", size=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Probability density", size=12)
    fig.suptitle(f"Stage 1 signal check — genuine vs impostor SSD ({n} subjects, exhaustive impostor pairs)", size=13)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
