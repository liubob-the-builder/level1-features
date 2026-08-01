#!/usr/bin/env python3
"""Column-wise bit density + DFT binning feature extraction for iris codes.

Pipeline:
  1. Collapse the 32 radial rows into a single 512-element angular density
     profile: density(θ) = fraction of valid bits that are 1 at column θ.
  2. Apply a real DFT to the density profile.
  3. Take the power spectrum (|X[f]|²) for positive frequencies f = 1..256,
     discarding DC (f=0, which is just global mean density, not texture).
  4. Bin the 256 power values into 16 log-spaced frequency bands and
     average within each band.

Result: a 16-dimensional float vector.

Rotation invariance: a circular column shift of the iris code shifts the
angular density profile cyclically, which only changes the phase of each DFT
coefficient — magnitudes (and therefore powers) are unchanged.
"""

import argparse
import numpy as np
from pathlib import Path

__all__ = [
    "column_density_profile",
    "dft_power_bins",
    "extract_dft_vector",
    "compute_ssd",
    "save_dft_vector",
    "load_dft_vector",
    "batch_extract",
]

N_BINS = 16
# Positive frequency indices: 1..256 for a 512-point real DFT
FREQ_MIN, FREQ_MAX = 1, 256


def _log_bin_edges(n_bins: int = N_BINS) -> np.ndarray:
    """Compute n_bins+1 integer-valued log-spaced bin edges for f=1..FREQ_MAX.

    Using np.logspace with floating-point edges creates empty bins at low
    frequencies: the gap between f=1 and f=2 (distance 1) spans ~2 bin-widths
    at ratio ≈1.41, so no integer falls in the intermediate bin.

    Fix: compute edges as rounded integers from geomspace, then take unique
    values. For n_bins=16 this yields edges
        [0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256]
    where consecutive upper bounds have ratio ≈1.41 and every bin is non-empty.
    """
    raw = np.round(np.geomspace(FREQ_MIN, FREQ_MAX, n_bins + 1)).astype(int)
    unique_upper = np.unique(raw)           # may be < n_bins+1 values if some bins merge
    return np.concatenate([[0], unique_upper])


# Precompute edges and bin assignment for each frequency index 1..256
_EDGES = _log_bin_edges(N_BINS)
_FREQ_INDICES = np.arange(FREQ_MIN, FREQ_MAX + 1)                  # 1..256
_BIN_ASSIGNMENT = np.searchsorted(_EDGES[1:], _FREQ_INDICES)       # 0-indexed bin


def column_density_profile(template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute the angular bit-density profile.

    For each column θ:
        density(θ) = Σ_r m(r,θ) · b(r,θ)  /  Σ_r m(r,θ)

    Columns where all bits are masked (Σ_r m = 0) are assigned density 0.5
    (neutral / uninformative) rather than NaN so the DFT stays well-defined.

    Returns:
        Float64 array of shape (n_cols,), values in [0, 1].
    """
    valid_counts = mask.sum(axis=0).astype(np.float64)     # (n_cols,)
    bit_counts   = (template * mask).sum(axis=0).astype(np.float64)

    # Avoid division by zero for fully-masked columns
    with np.errstate(invalid='ignore', divide='ignore'):
        density = np.where(valid_counts > 0, bit_counts / valid_counts, 0.5)

    return density


def dft_power_bins(density: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Compute 16 log-spaced DFT power bins from a density profile.

    Steps:
      - Real DFT of the density profile (length N = 512)
      - Power spectrum: |X[f]|² for positive frequencies f = 1..N//2
      - Group into n_bins log-spaced bins; each bin = mean power in that band

    Returns:
        Float64 array of shape (n_bins,).
    """
    N = len(density)
    # rfft gives coefficients for f = 0, 1, ..., N//2
    X = np.fft.rfft(density)

    # Power for positive frequencies 1..N//2 (discard DC at index 0)
    power = (np.abs(X[FREQ_MIN : FREQ_MAX + 1]) ** 2)   # shape (256,)

    # Bin edges and assignment (recompute if density length differs from 512)
    if N == 512:
        edges = _EDGES
        bin_ids = _BIN_ASSIGNMENT
    else:
        edges = _log_bin_edges(n_bins)
        freq_idx = np.arange(1, N // 2 + 1)
        bin_ids = np.searchsorted(edges[1:], freq_idx)
        power = (np.abs(X[1:]) ** 2)

    bins = np.zeros(n_bins)
    for b in range(n_bins):
        mask_b = bin_ids == b
        if mask_b.any():
            bins[b] = power[mask_b].mean()

    return bins


def extract_dft_vector(
    template: np.ndarray,
    mask: np.ndarray,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """Extract the 16-dim DFT power bin feature vector from an iris code.

    Args:
        template: Binary array of shape (n_rows, n_cols).
        mask:     Binary validity mask of same shape (1 = valid, 0 = masked).
        n_bins:   Number of log-spaced frequency bins (default 16).

    Returns:
        Float64 array of shape (n_bins,).
    """
    density = column_density_profile(template, mask)
    return dft_power_bins(density, n_bins=n_bins)


def compute_ssd(v1: np.ndarray, v2: np.ndarray) -> float:
    """Sum of squared differences between two feature vectors."""
    diff = v1 - v2
    return float(np.dot(diff, diff))


def save_dft_vector(vector: np.ndarray, path: Path) -> None:
    """Save a DFT feature vector to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


def load_dft_vector(path: Path) -> np.ndarray:
    """Load a DFT feature vector from a .npy file."""
    return np.load(Path(path))


def batch_extract(directory: Path, n_bins: int = N_BINS, verbose: bool = False) -> int:
    """Extract DFT vectors for all templates found under directory.

    Saves each vector as <stem>_dft.npy alongside the template file.

    Returns the number of vectors written.
    """
    template_files = sorted(Path(directory).glob("**/*_template.txt"))
    for tpath in template_files:
        stem = tpath.stem.replace("_template", "")
        subject_dir = tpath.parent
        with open(tpath) as f:
            template = np.array([list(r.strip()) for r in f], dtype=np.uint8)
        with open(subject_dir / f"{stem}_mask.txt") as f:
            mask = np.array([list(r.strip()) for r in f], dtype=np.uint8)
        vector = extract_dft_vector(template, mask, n_bins=n_bins)
        out_path = subject_dir / f"{stem}_dft.npy"
        save_dft_vector(vector, out_path)
        if verbose:
            print(f"  {subject_dir.name}/{stem}_dft.npy")
    return len(template_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract column-density DFT feature vectors from iris codes."
    )
    parser.add_argument(
        "-d", "--directory", type=Path, default=Path("generated"),
        help="root directory containing subject folders (default: generated)",
    )
    parser.add_argument(
        "--bins", type=int, default=N_BINS,
        help=f"number of log-spaced DFT power bins (default: {N_BINS})",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Extracting DFT vectors ({args.bins} log-spaced bins) from '{args.directory}' ...")
    n = batch_extract(args.directory, n_bins=args.bins, verbose=args.verbose)
    sample = next(Path(args.directory).glob("**/*_dft.npy"), None)
    dim = load_dft_vector(sample).shape[0] if sample else "?"
    print(f"Done. Wrote {n} vectors of dimension {dim}.")
