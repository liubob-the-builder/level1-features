#!/usr/bin/env python3
"""Angular DFT-magnitude spectrum for iris codes (Stage 1 spectrum inspection).

For each row of a binary iris code (length 512, treated as circular), take the
1D real FFT along the angular axis and its magnitude. Aggregate the 32
per-row magnitude spectra into one spectrum per code, either by averaging
(default) or by concatenating each row's low-frequency bins (kept as a flag
so the two aggregation strategies can be compared later if averaging turns
out to wash out the signal).

This is purely for spectrum inspection ahead of choosing which frequency
bins to keep for the Level-1 feature vector; no feature vector or bin
selection is built here.
"""

import argparse
from pathlib import Path
# Anchor to repo layout regardless of working directory:
# this file is in level1-features/experiments/ ; galleries are in ../../sic-gen/
_SICGEN = Path(__file__).resolve().parent.parent.parent / "sic-gen"
from typing import Literal

import numpy as np

__all__ = [
    "row_magnitude_spectrum",
    "code_magnitude_spectrum",
    "load_template",
    "batch_code_spectra",
]

N_COLS = 512
N_FREQS = N_COLS // 2 + 1  # rfft output length for N=512 -> 257 (includes DC at index 0)


def row_magnitude_spectrum(row: np.ndarray) -> np.ndarray:
    """1D real FFT magnitude spectrum of a single row (circular, length N_COLS).

    Returns:
        Float64 array of shape (N_COLS // 2 + 1,), magnitude |X[f]| for
        f = 0..N_COLS//2 (index 0 is the DC term).
    """
    X = np.fft.rfft(row.astype(np.float64))
    return np.abs(X)


MIN_VALID_FRAC = 0.5  # mask-aware mode: rows below this valid fraction are dropped from the average


def code_magnitude_spectrum(
    template: np.ndarray,
    row_agg: Literal["average", "concat"] = "average",
    n_low_bins: int = 32,
    mask: np.ndarray = None,
    min_valid_frac: float = MIN_VALID_FRAC,
) -> np.ndarray:
    """Aggregate per-row magnitude spectra into one spectrum (or vector) per code.

    Args:
        template: Binary array of shape (n_rows, N_COLS).
        row_agg:  'average'  -> mean magnitude spectrum across rows, shape (N_FREQS,).
                  'concat'   -> concatenate each row's lowest n_low_bins frequency
                                bins (bins 1..n_low_bins, DC excluded per row),
                                shape (n_rows * n_low_bins,).
        n_low_bins: number of low-frequency bins kept per row in 'concat' mode.
        mask: Optional binary array of same shape as template (1 = valid,
              0 = occluded). Default None reproduces the exact mask-blind
              behavior used for all existing synthetic (SIC-Gen) results:
              every row is FFT'd as-is, occluded bits included.
              If given, a pragmatic two-part scheme is used since an FFT
              cannot simply skip missing samples:
                (a) a row is dropped from the aggregate entirely if its
                    valid fraction is below `min_valid_frac` (default 0.5,
                    i.e. majority-occluded rows are excluded rather than
                    let a mean-filled row dominated by filler values distort
                    the spectrum);
                (b) for rows that are kept, occluded bits are replaced with
                    the mean of that row's own valid bits before the FFT
                    (mean-fill), so the DC term is undisturbed and no sharp
                    edges are introduced at occlusion boundaries.
              NOTE: in 'concat' mode, dropping rows changes the output
              dimension per code, which breaks fixed-length SSD comparison
              across codes -- 'concat' + mask is therefore not supported
              (raises ValueError). Use 'average' with a mask.
        min_valid_frac: valid-fraction threshold below which a row is
              dropped (mask-aware mode only). Ignored if mask is None.

    Returns:
        Float64 array; shape depends on row_agg (see above).
    """
    n_rows = template.shape[0]

    if mask is None:
        per_row = np.stack([row_magnitude_spectrum(template[r]) for r in range(n_rows)])  # (n_rows, N_FREQS)
    else:
        if row_agg == "concat":
            raise ValueError("mask-aware mode is not supported with row_agg='concat' "
                              "(row-dropping would produce a variable-length vector)")
        kept_spectra = []
        for r in range(n_rows):
            valid = mask[r].astype(bool)
            valid_frac = valid.mean() if len(valid) else 0.0
            if valid_frac < min_valid_frac:
                continue
            row = template[r].astype(np.float64).copy()
            if valid_frac < 1.0:
                row[~valid] = row[valid].mean()
            kept_spectra.append(row_magnitude_spectrum(row))
        if not kept_spectra:
            # Degenerate case: every row failed the threshold. Fall back to
            # all rows, mean-filled where possible, so a spectrum is still returned.
            kept_spectra = [row_magnitude_spectrum(template[r]) for r in range(n_rows)]
        per_row = np.stack(kept_spectra)  # (n_kept_rows, N_FREQS)

    if row_agg == "average":
        return per_row.mean(axis=0)  # (N_FREQS,)
    elif row_agg == "concat":
        return per_row[:, 1 : n_low_bins + 1].ravel()  # (n_rows * n_low_bins,)
    else:
        raise ValueError(f"unknown row_agg: {row_agg!r}")


def load_template(subject_dir: Path, stem: str) -> np.ndarray:
    """Load a binary template file (e.g. stem='1' for IC1, '2' for IC2)."""
    with open(subject_dir / f"{stem}_template.txt") as f:
        return np.array([list(r.strip()) for r in f], dtype=np.uint8)


def batch_code_spectra(
    directory: Path,
    stem: str = "1",
    row_agg: Literal["average", "concat"] = "average",
) -> tuple[np.ndarray, list[Path]]:
    """Compute the averaged magnitude spectrum for every subject's code `stem` under directory.

    Args:
        directory: root directory containing per-subject subfolders.
        stem:      '1' for IC1 (gallery), '2' for IC2 (probe).
        row_agg:   aggregation mode, see code_magnitude_spectrum.

    Returns:
        (spectra, subject_dirs) where spectra has shape (n_subjects, N_FREQS)
        for row_agg='average', sorted by subject directory name (numeric).
    """
    subject_dirs = sorted(
        (d for d in Path(directory).iterdir() if d.is_dir() and (d / f"{stem}_template.txt").exists()),
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )
    spectra = []
    for d in subject_dirs:
        template = load_template(d, stem)
        spectra.append(code_magnitude_spectrum(template, row_agg=row_agg))
    return np.stack(spectra), subject_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-code angular DFT-magnitude spectra (spectrum inspection only)."
    )
    parser.add_argument("-d", "--directory", type=Path, default=_SICGEN / "sicgen_gallery_300subjects")
    parser.add_argument("--stem", default="1", help="'1' for IC1 (gallery), '2' for IC2 (probe)")
    parser.add_argument("--row-agg", choices=["average", "concat"], default="average")
    args = parser.parse_args()

    spectra, subject_dirs = batch_code_spectra(args.directory, stem=args.stem, row_agg=args.row_agg)
    print(f"Computed spectra for {len(subject_dirs)} codes, shape {spectra.shape}")
