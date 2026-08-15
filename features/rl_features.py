#!/usr/bin/env python3
"""Run-length histogram feature extraction for iris codes.

Implements the Level-1 run-length histogram features from:
  Karakosta & Knottenbelt, "Iris Through the Looking Glass:
  Filter-Based Privacy-Preserving Identification".

For each row of a binary iris code, we count runs of consecutive 0s and 1s.
Runs of length l fall in bin (l-1); runs of length >= C fall in the last bin.
This gives a 2C-element histogram per row, and a (n_rows * 2C)-element
feature vector for the full iris code. With C=9 and 32 rows: 576 dimensions.

The feature is rotation-tolerant: a circular column shift rearranges run
boundaries but preserves the overall length distribution across a row.

Distance metric: sum of squared differences (SSD), matching the paper's
encrypted Level-1 comparison operator.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Optional

__all__ = [
    "row_rl_histogram",
    "extract_rl_vector",
    "compute_ssd",
    "save_rl_vector",
    "load_rl_vector",
]


def row_rl_histogram(row: np.ndarray, C: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the 2C-element run-length histogram for a single binary row.

    First C elements count 0-runs by length; next C count 1-runs by length.
    Bin index = min(run_length - 1, C - 1), so the last bin accumulates all
    runs of length >= C.

    Args:
        mask: Optional binary array of same shape as row (1 = valid, 0 =
              occluded). If None (default), the row is treated as fully
              valid -- current mask-blind behavior, unchanged. If given,
              runs are computed only within maximal contiguous runs of
              valid bits: an occluded bit terminates whatever run precedes
              it and is not itself counted, and no run crosses an occluded
              gap. Each valid segment is handled independently (no
              wraparound across a gap or across the row boundary), matching
              the mask-blind path's existing non-circular boundary handling.
    """
    if mask is None:
        hist = np.zeros(2 * C, dtype=np.int32)
        if len(row) == 0:
            return hist
        changes = np.where(np.diff(row.astype(np.int8)))[0] + 1
        boundaries = np.concatenate(([0], changes, [len(row)]))
        run_lengths = np.diff(boundaries)
        run_values = row[boundaries[:-1]]          # value (0 or 1) of each run
        bin_indices = np.minimum(run_lengths - 1, C - 1) + run_values.astype(np.int32) * C
        np.add.at(hist, bin_indices, 1)
        return hist

    hist = np.zeros(2 * C, dtype=np.int32)
    valid = mask.astype(bool)
    if not valid.any():
        return hist
    seg_changes = np.where(np.diff(valid.astype(np.int8)))[0] + 1
    seg_boundaries = np.concatenate(([0], seg_changes, [len(valid)]))
    for start, end in zip(seg_boundaries[:-1], seg_boundaries[1:]):
        if valid[start]:
            hist += row_rl_histogram(row[start:end], C)
    return hist


def extract_rl_vector(template: np.ndarray, C: int = 9, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract the run-length histogram feature vector from an iris code.

    Args:
        template: Binary array of shape (n_rows, n_cols).
        C:        Number of histogram bins. Default 9 → 576-dim vector for
                  32-row iris codes (matching the paper's stated dimensionality).
                  Use C=6 for the 384-dim variant if preferred.
        mask:     Optional binary array of same shape as template (1 = valid,
                  0 = occluded). Default None reproduces the exact mask-blind
                  behavior used for all existing synthetic (SIC-Gen) results.
                  If given, occluded bits are excluded from run-length
                  counting per row (see row_rl_histogram) -- this applies
                  identically regardless of C, so both RL-C9 and RL-C6
                  become mask-aware from this single change.

    Returns:
        Integer array of shape (n_rows * 2C,).
    """
    if mask is None:
        return np.concatenate([row_rl_histogram(row, C) for row in template])
    return np.concatenate([row_rl_histogram(row, C, mask=m) for row, m in zip(template, mask)])


def compute_ssd(v1: np.ndarray, v2: np.ndarray) -> float:
    """Sum of squared differences between two feature vectors.

    Lower SSD indicates more similar iris codes.
    This mirrors the encrypted Level-1 comparison from the paper:
        SSD(fp, fi) = Σ_j (fp[j] - fi[j])²
    """
    diff = v1.astype(np.float64) - v2.astype(np.float64)
    return float(np.dot(diff, diff))


def save_rl_vector(vector: np.ndarray, path: Path) -> None:
    """Save a run-length feature vector to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


def load_rl_vector(path: Path) -> np.ndarray:
    """Load a run-length feature vector from a .npy file."""
    return np.load(Path(path))


def _extract_and_save(template_path: Path, C: int) -> None:
    """Load a template file and save its RL vector alongside it."""
    template_path = Path(template_path)
    with open(template_path) as f:
        template = np.array([list(row.strip()) for row in f.readlines()], dtype=np.uint8)
    vector = extract_rl_vector(template, C)
    stem = template_path.stem.replace("_template", "")
    out_path = template_path.parent / f"{stem}_rl.npy"
    save_rl_vector(vector, out_path)


def batch_extract(directory: Path, C: int = 9, verbose: bool = False) -> int:
    """Extract RL vectors for all templates found under directory.

    Looks for files matching the pattern *_template.txt (the naming convention
    used by sic-gen). Saves each vector as <stem>_rl.npy in the same folder.

    Returns the number of vectors written.
    """
    template_files = sorted(Path(directory).glob("**/*_template.txt"))
    for tpath in template_files:
        _extract_and_save(tpath, C)
        if verbose:
            stem = tpath.stem.replace("_template", "")
            print(f"  {tpath.parent.name}/{stem}_rl.npy")
    return len(template_files)


def rank_gallery(
    probe_path: Path,
    gallery_dir: Path,
    C: int = 9,
    top_k: Optional[int] = None,
) -> list[tuple[float, Path]]:
    """Rank gallery entries against a probe using SSD on RL vectors.

    Probe RL vector is computed on-the-fly from probe_path (a *_template.txt
    file) if a corresponding *_rl.npy does not exist yet.

    Args:
        probe_path:  Path to the probe *_rl.npy (or *_template.txt).
        gallery_dir: Root directory containing subject sub-folders with *_rl.npy.
        C:           Histogram bin count (must match what was used when saving).
        top_k:       If set, return only the top-k closest candidates.

    Returns:
        List of (ssd, rl_vector_path) sorted ascending by SSD (best first).
    """
    probe_path = Path(probe_path)
    if probe_path.suffix == ".txt":
        with open(probe_path) as f:
            template = np.array([list(row.strip()) for row in f.readlines()], dtype=np.uint8)
        probe_vec = extract_rl_vector(template, C)
    else:
        probe_vec = load_rl_vector(probe_path)

    gallery_files = sorted(Path(gallery_dir).glob("**/*_rl.npy"))
    scores = [(compute_ssd(probe_vec, load_rl_vector(g)), g) for g in gallery_files]
    scores.sort(key=lambda x: x[0])
    return scores[:top_k] if top_k is not None else scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract run-length histogram feature vectors from iris codes."
    )
    parser.add_argument(
        "-d", "--directory", type=Path, default=Path("generated"),
        help="root directory containing subject folders (default: generated)",
    )
    parser.add_argument(
        "-C", "--bins", type=int, default=9,
        help="number of histogram bins per run type (default: 9 → 576-dim for 32-row codes)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Extracting RL vectors (C={args.bins}) from '{args.directory}' ...")
    n = batch_extract(args.directory, C=args.bins, verbose=args.verbose)
    dim = None
    sample = next(Path(args.directory).glob("**/*_rl.npy"), None)
    if sample:
        dim = load_rl_vector(sample).shape[0]
    print(f"Done. Wrote {n} vectors" + (f" of dimension {dim}." if dim else "."))
