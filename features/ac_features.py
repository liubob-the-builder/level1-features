#!/usr/bin/env python3
"""Autocorrelation lag-energy feature extraction for iris codes.

Implements the Level-1 autocorrelation features from:
  Karakosta & Knottenbelt, "Iris Through the Looking Glass:
  Filter-Based Privacy-Preserving Identification".

Two aggregation modes are provided:
  'concat'  -> 5 energy bins x n_rows = 160 dims for a 32-row code
  'mean'    -> validity-weighted average across rows = 5 dims

The paper states 90 dims, which does not follow cleanly from
"5 bins x 32 rows" under any described aggregation. Both variants
above are implemented so recall performance of each can be compared.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Literal

__all__ = [
    "row_autocorrelation",
    "lag_energy_bins",
    "extract_ac_vector",
    "compute_ssd",
    "save_ac_vector",
    "load_ac_vector",
    "batch_extract",
]

# Lag ranges from the paper (inclusive on both ends).
# Each range captures a different scale of iris texture structure:
#   1-2  : texture fineness  (adjacent bit agreement — captures run continuity)
#   3-5  : local grain       (short-range oscillation within a run cluster)
#   6-10 : Gabor-scale periodicity (matches the ~6.5 bit mean run length in SIC-Gen)
#   11-20: pattern repetition
#   21-40: global structure  (long-range periodicity across the full row)
LAG_GROUPS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40)]
MAX_LAG = 40


def row_autocorrelation(row: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute the masked circular autocorrelation for lags 1..MAX_LAG.

    For each lag ℓ, the formula is:
        R(ℓ) = Σ_θ m(θ) m(θ+ℓ) x(θ) x(θ+ℓ)  /  Σ_θ m(θ) m(θ+ℓ)
    where x(θ) = 2b(θ) - 1 (bipolar) and summation wraps circularly.

    If the denominator is zero (no mutually valid pairs), R(ℓ) = 0.

    Returns:
        Array of shape (MAX_LAG,) containing R(1), R(2), ..., R(MAX_LAG).
    """
    x = 2.0 * row.astype(np.float64) - 1.0   # {0,1} -> {-1,+1}
    m = mask.astype(np.float64)

    R = np.empty(MAX_LAG)
    for lag in range(1, MAX_LAG + 1):
        # Circular shift by 'lag' positions along the row
        m_shifted = np.roll(m, -lag)
        x_shifted = np.roll(x, -lag)

        joint_valid = m * m_shifted           # 1 where both positions are valid
        denom = joint_valid.sum()

        if denom > 0:
            R[lag - 1] = np.dot(joint_valid * x, x_shifted) / denom
        else:
            R[lag - 1] = 0.0

    return R


def lag_energy_bins(R: np.ndarray) -> np.ndarray:
    """Summarise an autocorrelation vector into 5 lag-energy bins.

    E_k = (1 / |L_k|)  Σ_{ℓ ∈ L_k}  R(ℓ)²

    Squaring converts signed correlations to energies (always >= 0),
    and captures how much periodicity exists at each scale regardless
    of whether the phase is positive or negative.

    Returns:
        Array of shape (5,).
    """
    bins = np.empty(len(LAG_GROUPS))
    for k, (lo, hi) in enumerate(LAG_GROUPS):
        # R is 0-indexed: lag ℓ -> R[ℓ-1]
        bins[k] = np.mean(R[lo - 1 : hi] ** 2)
    return bins


def extract_ac_vector(
    template: np.ndarray,
    mask: np.ndarray,
    mode: Literal["concat", "mean"] = "concat",
) -> np.ndarray:
    """Extract autocorrelation lag-energy feature vector from an iris code.

    For each row:
      1. Compute masked circular autocorrelation R(ℓ) for ℓ = 1..40.
      2. Summarise into 5 lag-energy bins.

    Then aggregate across all rows according to 'mode':
      'concat' -> concatenate per-row 5-dim vectors -> shape (n_rows * 5,)
                  For a 32-row code: 160 dims.
      'mean'   -> validity-weighted average across rows -> shape (5,)
                  Weight for row r = fraction of valid bits in that row.

    Args:
        template: Binary array of shape (n_rows, n_cols).
        mask:     Binary array of same shape (1 = valid, 0 = occluded).
        mode:     'concat' (160-dim) or 'mean' (5-dim).

    Returns:
        Float64 array of the described shape.
    """
    n_rows = template.shape[0]
    per_row_energy = np.empty((n_rows, 5))
    row_validity = np.empty(n_rows)

    for r in range(n_rows):
        R = row_autocorrelation(template[r], mask[r])
        per_row_energy[r] = lag_energy_bins(R)
        row_validity[r] = mask[r].mean()        # fraction of valid bits

    if mode == "concat":
        return per_row_energy.ravel()           # (n_rows * 5,) = 160 for 32-row codes

    # mode == "mean": validity-weighted average across rows
    total_weight = row_validity.sum()
    if total_weight == 0:
        weights = np.ones(n_rows) / n_rows
    else:
        weights = row_validity / total_weight
    return (per_row_energy * weights[:, None]).sum(axis=0)   # (5,)


def compute_ssd(v1: np.ndarray, v2: np.ndarray) -> float:
    """Sum of squared differences between two feature vectors.

    Lower SSD indicates more similar iris codes. Identical to the metric
    in rl_features.py; re-exported here for convenience.
    """
    diff = v1 - v2
    return float(np.dot(diff, diff))


def save_ac_vector(vector: np.ndarray, path: Path) -> None:
    """Save an autocorrelation feature vector to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


def load_ac_vector(path: Path) -> np.ndarray:
    """Load an autocorrelation feature vector from a .npy file."""
    return np.load(Path(path))


def _load_template_and_mask(subject_dir: Path, stem: str):
    """Load template and mask arrays for one iris code."""
    with open(subject_dir / f"{stem}_template.txt") as f:
        template = np.array([list(r.strip()) for r in f], dtype=np.uint8)
    with open(subject_dir / f"{stem}_mask.txt") as f:
        mask = np.array([list(r.strip()) for r in f], dtype=np.uint8)
    return template, mask


def batch_extract(
    directory: Path,
    mode: Literal["concat", "mean"] = "concat",
    verbose: bool = False,
) -> int:
    """Extract AC vectors for all templates found under directory.

    Saves each vector as <stem>_ac_<mode>.npy in the same folder as
    the corresponding template file.

    Returns the number of vectors written.
    """
    template_files = sorted(Path(directory).glob("**/*_template.txt"))
    for tpath in template_files:
        stem = tpath.stem.replace("_template", "")
        subject_dir = tpath.parent
        template, mask = _load_template_and_mask(subject_dir, stem)
        vector = extract_ac_vector(template, mask, mode=mode)
        out_name = f"{stem}_ac_{mode}.npy"
        save_ac_vector(vector, subject_dir / out_name)
        if verbose:
            print(f"  {subject_dir.name}/{out_name}")
    return len(template_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract autocorrelation lag-energy feature vectors from iris codes."
    )
    parser.add_argument(
        "-d", "--directory", type=Path, default=Path("generated"),
        help="root directory containing subject folders (default: generated)",
    )
    parser.add_argument(
        "--mode", choices=["concat", "mean", "both"], default="both",
        help="aggregation mode: concat (160-dim), mean (5-dim), or both (default: both)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    modes = ["concat", "mean"] if args.mode == "both" else [args.mode]
    for mode in modes:
        print(f"Extracting AC vectors (mode={mode}) from '{args.directory}' ...")
        n = batch_extract(args.directory, mode=mode, verbose=args.verbose)
        sample = next(Path(args.directory).glob(f"**/*_ac_{mode}.npy"), None)
        dim = load_ac_vector(sample).shape[0] if sample else "?"
        print(f"  Done. Wrote {n} vectors of dimension {dim}.")
