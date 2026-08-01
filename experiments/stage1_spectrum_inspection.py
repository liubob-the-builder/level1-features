#!/usr/bin/env python3
"""Stage 1 — spectrum inspection for the angular DFT-magnitude Level-1 feature.

For each of the 300 gallery codes (IC1) in sicgen_gallery_300subjects/:
  1. Real FFT of each of the 32 rows along the 512-length angular axis.
  2. Magnitude of each row's spectrum.
  3. Average the 32 per-row magnitude spectra -> one spectrum per code.
  4. Drop bin 0 (DC).

Across the 300 codes, plot:
  (a) the mean magnitude spectrum, and
  (b) the per-bin variance across codes.

This script only inspects the spectrum to inform bin selection — it does not
build a feature vector or run any scoring.
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
OUT_PATH = Path("stage1_spectrum_inspection.png")

# Reference palette (dataviz skill): sequential blue for magnitude, aqua for variance.
COLOR_MEAN = "#2a78d6"
COLOR_VAR = "#1baf7a"


def main():
    spectra, subject_dirs = batch_code_spectra(GALLERY_DIR, stem="1", row_agg="average")
    print(f"Loaded {len(subject_dirs)} IC1 codes, spectrum shape per code: {spectra.shape[1]} bins (incl. DC)")

    # Drop bin 0 (DC)
    spectra = spectra[:, 1:]  # (300, 256)
    freq_bins = np.arange(1, spectra.shape[1] + 1)  # 1..256

    mean_spectrum = spectra.mean(axis=0)
    var_spectrum = spectra.var(axis=0)

    fig, (ax_mean, ax_var) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax_mean.plot(freq_bins, mean_spectrum, color=COLOR_MEAN, linewidth=1.5)
    ax_mean.set_ylabel("Mean magnitude", size=12)
    ax_mean.set_title(f"(a) Mean angular DFT magnitude spectrum — averaged across {len(subject_dirs)} IC1 codes", size=13)
    ax_mean.grid(True, alpha=0.3)
    ax_mean.margins(x=0.01)

    ax_var.plot(freq_bins, var_spectrum, color=COLOR_VAR, linewidth=1.5)
    ax_var.set_xlabel("Frequency bin (DC dropped)", size=12)
    ax_var.set_ylabel("Variance across codes", size=12)
    ax_var.set_title(f"(b) Per-bin variance across {len(subject_dirs)} IC1 codes", size=13)
    ax_var.grid(True, alpha=0.3)
    ax_var.margins(x=0.01)

    fig.suptitle("Stage 1 spectrum inspection — sicgen_gallery_300subjects (row_agg=average)", size=14)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {OUT_PATH}")

    # Also save the raw arrays for reference / later reuse when picking bins.
    np.savez(
        "stage1_spectrum_inspection.npz",
        freq_bins=freq_bins,
        mean_spectrum=mean_spectrum,
        var_spectrum=var_spectrum,
        all_spectra=spectra,
    )
    print("Saved raw spectra to stage1_spectrum_inspection.npz")

    # Print the top bins by variance, to make discussion easier.
    top_var_idx = np.argsort(var_spectrum)[::-1][:20]
    print("\nTop 20 bins by cross-code variance (bin: variance, mean):")
    for idx in top_var_idx:
        print(f"  bin {freq_bins[idx]:>3}: var={var_spectrum[idx]:.4f}  mean={mean_spectrum[idx]:.4f}")


if __name__ == "__main__":
    main()
