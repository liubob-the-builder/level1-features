#!/usr/bin/env python3
"""Row-redundancy check: are a code's rows near-copies of each other?

Hypothesis under test: SIC-Gen builds a single "barcode" row and duplicates
it down the code before perturbing it, so its 32 rows are far more mutually
redundant than real iris-code rows -- which would explain why row-aggregating
Level-1 features (DFT, AC, RL) score much higher on SIC-Gen than they would
on real CASIA-Iris-Thousand data.

Four measurements, computed per code and reported as a distribution across
the whole gallery:
  1. Within-code row similarity, bit domain: mean pairwise Hamming distance
     between all C(n_rows,2) row pairs. ~0.5 = independent rows, ~0 = near-
     copies. Also an alignment-tolerant version allowing +/-7 circular shifts
     per pair (min HD over shifts), in case rows are shifted rather than
     exact copies.
  2. Within-code row similarity, spectral domain: mean pairwise Pearson
     correlation between rows' DFT magnitude spectra (DC dropped). ~1.0 =
     rows share the same spectral signature; ~0 = independent textures.
  3. Effective row count via PCA, on both representations:
       (a) the raw n_rows x n_cols bit matrix
       (b) the n_rows x (n_cols//2) per-row magnitude-spectrum matrix
     reporting how many components explain 95% of variance in each. ~1-2 =
     rows are essentially one repeated pattern; ~n_rows = independent rows.
     Comparing (a) vs (b) is the key check: if the bit matrix says rows are
     distinct but the spectral matrix says they're near-identical, rows are
     shifted/perturbed copies -- different in bit position, redundant in
     structure. Since Level-1 features (DFT/AC/RL) read structure, not raw
     bit position, that's the redundancy that would matter to them.

Reusable by design: `load_templates()` and `analyze_directory()` accept any
directory and don't hardcode 32x512. Point --directory at a different code
set (e.g. a real CASIA-Iris-Thousand export) with a matching --stem/--pattern
to get the identical measurements for a synthetic-vs-real comparison -- the
synthetic-only numbers here are not meaningful without that baseline.
"""

import argparse
from pathlib import Path
# Anchor to repo layout regardless of working directory:
# this file is in level1-features/experiments/ ; galleries are in ../../sic-gen/
_SICGEN = Path(__file__).resolve().parent.parent.parent / "sic-gen"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MAX_SHIFT = 7
PCA_THRESHOLD = 0.95

COLOR_A = "#2a78d6"   # blue   -- unaligned / bit-domain
COLOR_B = "#1baf7a"   # aqua   -- aligned / spectral-domain


def load_templates(directory: Path, stem: str = "1"):
    """Load binary code matrices from `directory`.

    Two supported layouts:
      - SIC-Gen style: subject subdirs each containing `<stem>_template.txt`.
      - Flat style (fallback, e.g. a CASIA export): `directory` contains
        `*.txt` template files directly, one per code.

    Returns (names, templates) where templates is a list of uint8 arrays,
    each of shape (n_rows, n_cols). n_rows/n_cols need not be 32/512.
    """
    directory = Path(directory)
    subject_dirs = sorted(
        (d for d in directory.iterdir() if d.is_dir() and (d / f"{stem}_template.txt").exists()),
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )
    if subject_dirs:
        names, templates = [], []
        for d in subject_dirs:
            with open(d / f"{stem}_template.txt") as f:
                t = np.array([list(r.strip()) for r in f], dtype=np.uint8)
            names.append(d.name)
            templates.append(t)
        return names, templates

    files = sorted(directory.glob("*.txt"))
    names, templates = [], []
    for f in files:
        with open(f) as fh:
            t = np.array([list(r.strip()) for r in fh], dtype=np.uint8)
        names.append(f.stem)
        templates.append(t)
    return names, templates


def pairwise_hd_unaligned(template: np.ndarray) -> float:
    """Mean Hamming distance over all row pairs, no shifting."""
    n_rows = template.shape[0]
    diff = template[:, None, :] != template[None, :, :]
    hd = diff.mean(axis=2)
    iu = np.triu_indices(n_rows, k=1)
    return float(hd[iu].mean())


def pairwise_hd_aligned(template: np.ndarray, max_shift: int = MAX_SHIFT) -> float:
    """Mean Hamming distance over all row pairs, allowing +/-max_shift
    circular shifts per pair and taking the minimum HD over those shifts."""
    n_rows = template.shape[0]
    best = np.full((n_rows, n_rows), np.inf)
    for s in range(-max_shift, max_shift + 1):
        rolled = np.roll(template, shift=s, axis=1)
        diff = template[:, None, :] != rolled[None, :, :]
        hd = diff.mean(axis=2)
        best = np.minimum(best, hd)
    iu = np.triu_indices(n_rows, k=1)
    return float(best[iu].mean())


def spectral_matrix(template: np.ndarray) -> np.ndarray:
    """Per-row DFT magnitude spectrum, DC dropped: shape (n_rows, n_cols//2)."""
    X = np.fft.rfft(template.astype(np.float64), axis=1)
    return np.abs(X)[:, 1:]


def mean_pairwise_spectral_corr(spectra: np.ndarray) -> float:
    n_rows = spectra.shape[0]
    corr = np.corrcoef(spectra)
    iu = np.triu_indices(n_rows, k=1)
    return float(corr[iu].mean())


def pca_components_for_variance(matrix: np.ndarray, threshold: float = PCA_THRESHOLD) -> int:
    """Number of PCA components (on rows-as-samples) needed to reach
    `threshold` cumulative explained variance."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return 1
    s = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    var = s ** 2
    total = var.sum()
    if total == 0:
        return 1
    cum = np.cumsum(var) / total
    return int(np.searchsorted(cum, threshold) + 1)


def analyze_directory(directory: Path, stem: str = "1", max_shift: int = MAX_SHIFT, label: str = None) -> dict:
    names, templates = load_templates(directory, stem=stem)
    n = len(templates)
    if n == 0:
        raise ValueError(f"No codes found in {directory} (stem={stem!r})")
    n_rows = templates[0].shape[0]
    print(f"Loaded {n} codes from {directory} (stem={stem!r}), shape {templates[0].shape}")

    hd_unaligned = np.empty(n)
    hd_aligned = np.empty(n)
    spec_corr = np.empty(n)
    pca_bit = np.empty(n, dtype=int)
    pca_spec = np.empty(n, dtype=int)

    for i, t in enumerate(templates):
        hd_unaligned[i] = pairwise_hd_unaligned(t)
        hd_aligned[i] = pairwise_hd_aligned(t, max_shift=max_shift)
        spectra = spectral_matrix(t)
        spec_corr[i] = mean_pairwise_spectral_corr(spectra)
        pca_bit[i] = pca_components_for_variance(t.astype(np.float64))
        pca_spec[i] = pca_components_for_variance(spectra)

    return {
        "label": label or str(directory),
        "names": names,
        "n_codes": n,
        "n_rows": n_rows,
        "hd_unaligned": hd_unaligned,
        "hd_aligned": hd_aligned,
        "spec_corr": spec_corr,
        "pca_bit": pca_bit,
        "pca_spec": pca_spec,
    }


def print_report(result: dict) -> None:
    n_rows = result["n_rows"]
    print(f"\n{'=' * 78}")
    print(f"  {result['label']}  (n={result['n_codes']} codes, {n_rows} rows/code)")
    print(f"{'=' * 78}")

    hd_u, hd_a = result["hd_unaligned"], result["hd_aligned"]
    print(f"\n1. Within-code row Hamming distance (baseline: independent rows -> ~0.500)")
    print(f"   Unaligned:        mean={hd_u.mean():.4f}  std={hd_u.std():.4f}")
    print(f"   Aligned (+/-{MAX_SHIFT}):  mean={hd_a.mean():.4f}  std={hd_a.std():.4f}")

    sc = result["spec_corr"]
    print(f"\n2. Within-code row spectral correlation (baseline: independent textures -> ~0)")
    print(f"   mean={sc.mean():.4f}  std={sc.std():.4f}")

    pb, ps = result["pca_bit"], result["pca_spec"]
    print(f"\n3. Effective row count (# PCA components for {PCA_THRESHOLD:.0%} variance, max={n_rows})")
    print(f"   Bit-domain matrix:      mean={pb.mean():.2f}  std={pb.std():.2f}  median={np.median(pb):.1f}")
    print(f"   Spectral-domain matrix: mean={ps.mean():.2f}  std={ps.std():.2f}  median={np.median(ps):.1f}")
    print(f"   Mean gap (bit - spectral): {pb.mean() - ps.mean():+.2f} "
          f"(positive => rows differ in bit position but are more redundant in structure)")


def plot_report(result: dict, out_path: Path) -> None:
    n_rows = result["n_rows"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ((ax_hd, ax_spec), (ax_pca, ax_scatter)) = axes

    hd_bins = np.linspace(0, 0.5, 50)
    ax_hd.hist(result["hd_unaligned"], bins=hd_bins, density=True, alpha=0.75,
               color="white", edgecolor=COLOR_A, linewidth=1.5,
               label=f"Unaligned  μ={result['hd_unaligned'].mean():.3f}")
    ax_hd.hist(result["hd_aligned"], bins=hd_bins, density=True, alpha=0.75,
               color="white", edgecolor=COLOR_B, linewidth=1.5,
               label=f"Aligned ±{MAX_SHIFT}  μ={result['hd_aligned'].mean():.3f}")
    ax_hd.axvline(0.5, color="black", linestyle="--", linewidth=1, label="independent-rows baseline (0.5)")
    ax_hd.set_xlabel("Mean pairwise row Hamming distance")
    ax_hd.set_ylabel("Probability density")
    ax_hd.set_title("1. Within-code row similarity (bit domain)")
    ax_hd.legend(fontsize=8.5)
    ax_hd.grid(True, alpha=0.3)

    ax_spec.hist(result["spec_corr"], bins=50, density=True, alpha=0.75,
                 color="white", edgecolor=COLOR_A, linewidth=1.5,
                 label=f"μ={result['spec_corr'].mean():.3f}")
    ax_spec.set_xlabel("Mean pairwise row spectral correlation")
    ax_spec.set_ylabel("Probability density")
    ax_spec.set_title("2. Within-code row similarity (spectral domain)")
    ax_spec.legend(fontsize=8.5)
    ax_spec.grid(True, alpha=0.3)

    pca_bins = np.arange(0.5, n_rows + 1.5, 1)
    ax_pca.hist(result["pca_bit"], bins=pca_bins, density=True, alpha=0.75,
                color="white", edgecolor=COLOR_A, linewidth=1.5,
                label=f"Bit domain  μ={result['pca_bit'].mean():.1f}")
    ax_pca.hist(result["pca_spec"], bins=pca_bins, density=True, alpha=0.75,
                color="white", edgecolor=COLOR_B, linewidth=1.5,
                label=f"Spectral domain  μ={result['pca_spec'].mean():.1f}")
    ax_pca.set_xlabel(f"# PCA components for {PCA_THRESHOLD:.0%} variance (max {n_rows})")
    ax_pca.set_ylabel("Probability density")
    ax_pca.set_title("3. Effective row count (PCA)")
    ax_pca.legend(fontsize=8.5)
    ax_pca.grid(True, alpha=0.3)

    jitter = np.random.default_rng(0).uniform(-0.15, 0.15, size=result["n_codes"])
    ax_scatter.scatter(result["pca_bit"] + jitter, result["pca_spec"] + jitter,
                        s=8, alpha=0.15, color=COLOR_A, edgecolors="none")
    lim = [0, n_rows + 1]
    ax_scatter.plot(lim, lim, color="black", linestyle="--", linewidth=1, label="bit = spectral")
    ax_scatter.set_xlim(lim)
    ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel("PCA components, bit domain")
    ax_scatter.set_ylabel("PCA components, spectral domain")
    ax_scatter.set_title("Bit-domain vs spectral-domain effective row count")
    ax_scatter.legend(fontsize=8.5)
    ax_scatter.grid(True, alpha=0.3)

    fig.suptitle(f"Row redundancy check — {result['label']} (n={result['n_codes']})", size=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Row-redundancy check for iris codes.")
    parser.add_argument("-d", "--directory", type=Path, default=_SICGEN / "sicgen_gallery_2000subjects")
    parser.add_argument("--stem", default="1", help="'1' for IC1, '2' for IC2 (SIC-Gen layout only)")
    parser.add_argument("--max-shift", type=int, default=MAX_SHIFT)
    parser.add_argument("--label", default=None, help="label for print/plot titles (default: directory path)")
    parser.add_argument("--out-prefix", default="row_redundancy", help="output file prefix")
    args = parser.parse_args()

    result = analyze_directory(args.directory, stem=args.stem, max_shift=args.max_shift, label=args.label)
    print_report(result)
    plot_report(result, Path(f"{args.out_prefix}.png"))

    np.savez(
        f"{args.out_prefix}.npz",
        hd_unaligned=result["hd_unaligned"],
        hd_aligned=result["hd_aligned"],
        spec_corr=result["spec_corr"],
        pca_bit=result["pca_bit"],
        pca_spec=result["pca_spec"],
        n_rows=result["n_rows"],
    )
    print(f"Saved arrays to {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
