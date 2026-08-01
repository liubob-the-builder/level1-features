#!/usr/bin/env python3
"""Negative-control experiment: is DFT's Stage-2 performance an artifact of
SIC-Gen encoding subject identity through the empirical per-subject run
length, rather than genuine discriminative texture structure?

All measurements below are computed on IC1 (gallery) codes from
sicgen_gallery_2000subjects/, using the same subject ordering as
stage2_evaluate.py (sorted numeric subject dir names) so that indices line
up with the ranks saved in stage2_ranks.npz.

Four checks:
  1. Correlation: per-subject mean run length vs. DFT peak-bin position.
     Prediction under the artifact hypothesis: strongly negative, with
     peak_bin ~ 512 / (2 * mean_run_length).
  2. Confusability: for each probe, do the top-10 DFT-ranked *wrong*
     candidates have closer run lengths to the true subject than 10 random
     gallery subjects would?
  3. Failure clustering: do DFT's 60 failed probes (true rank > 50) sit in
     locally dense regions of the run-length distribution (i.e. regions
     with many mutually confusable subjects)?
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
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from dft_spectrum import code_magnitude_spectrum, load_template

GALLERY_DIR = _SICGEN / "sicgen_gallery_2000subjects"
DFT_BINS = np.arange(30, 51)  # the feature actually used in Stage 2
N_COLS = 512
FAILURE_RANK_THRESHOLD = 50
RNG_SEED = 0
N_RANDOM_REPS = 50  # repetitions of the 10-random-subject draw, averaged for a stable baseline

COLOR_ALL = "#2a78d6"    # blue -- population / all subjects
COLOR_FAIL = "#e34948"   # red  -- DFT failure subset


def circular_mean_run_length_row(row: np.ndarray) -> float:
    """Mean length of maximal runs of identical bits in a single row, treating
    the 512-column axis as circular (wraps around), matching the DFT's own
    circular treatment of the angular axis."""
    n = len(row)
    changes = np.where(np.diff(row.astype(np.int8)) != 0)[0] + 1
    if len(changes) == 0:
        return float(n)  # constant row: one run spanning the whole circle
    boundaries = np.concatenate(([0], changes, [n]))
    run_lengths = np.diff(boundaries).astype(np.float64)
    if row[0] == row[-1]:
        # first and last runs are actually one run that wraps around
        run_lengths[0] += run_lengths[-1]
        run_lengths = run_lengths[:-1]
    return float(run_lengths.mean())


def circular_mean_run_length_code(template: np.ndarray) -> float:
    """Per-code mean run length: average across the 32 rows of each row's
    circular mean run length."""
    return float(np.mean([circular_mean_run_length_row(row) for row in template]))


def peak_bin_and_magnitude(full_spectrum: np.ndarray) -> tuple[int, float]:
    """Argmax bin (excluding DC at index 0) and its magnitude."""
    idx = 1 + int(np.argmax(full_spectrum[1:]))
    return idx, float(full_spectrum[idx])


def ssd_matrix(gallery: np.ndarray, probe: np.ndarray) -> np.ndarray:
    g_sq = np.sum(gallery ** 2, axis=1)
    p_sq = np.sum(probe ** 2, axis=1)
    cross = probe @ gallery.T
    D = p_sq[:, None] + g_sq[None, :] - 2 * cross
    return np.maximum(D, 0.0)


def main():
    subject_dirs = sorted(
        (d for d in GALLERY_DIR.iterdir() if d.is_dir() and (d / "1_template.txt").exists()),
        key=lambda d: int(d.name),
    )
    n = len(subject_dirs)
    print(f"Loaded {n} subjects from {GALLERY_DIR}")

    # ---- Per-subject run length, DFT peak bin/magnitude, and DFT feature (bins 30-50) ----
    mean_run_length = np.empty(n)
    peak_bin = np.empty(n, dtype=int)
    peak_mag = np.empty(n)
    dft_gallery_feat = np.empty((n, len(DFT_BINS)))
    dft_probe_feat = np.empty((n, len(DFT_BINS)))

    for i, sd in enumerate(subject_dirs):
        ic1 = load_template(sd, "1")
        mean_run_length[i] = circular_mean_run_length_code(ic1)
        full_spec_ic1 = code_magnitude_spectrum(ic1, row_agg="average")
        peak_bin[i], peak_mag[i] = peak_bin_and_magnitude(full_spec_ic1)
        dft_gallery_feat[i] = full_spec_ic1[DFT_BINS]

        ic2 = load_template(sd, "2")
        full_spec_ic2 = code_magnitude_spectrum(ic2, row_agg="average")
        dft_probe_feat[i] = full_spec_ic2[DFT_BINS]

    print(f"Mean run length across population: {mean_run_length.mean():.3f} +/- {mean_run_length.std():.3f}")
    print(f"Peak bin across population: mean={peak_bin.mean():.2f}  mode-ish range covers "
          f"{np.percentile(peak_bin, 5):.0f}-{np.percentile(peak_bin, 95):.0f} (5th-95th pct)")

    # ================================================================
    # 1. Correlation test: mean run length vs DFT peak bin
    # ================================================================
    pearson_r, pearson_p = pearsonr(mean_run_length, peak_bin)
    spearman_rho, spearman_p = spearmanr(mean_run_length, peak_bin)
    print("\n--- 1. Correlation: mean run length vs DFT peak bin ---")
    print(f"  Pearson  r   = {pearson_r:.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman rho = {spearman_rho:.4f}  (p={spearman_p:.2e})")

    # ================================================================
    # 2. Confusability test
    # ================================================================
    D = ssd_matrix(dft_gallery_feat, dft_probe_feat)
    order = np.argsort(D, axis=1)  # ascending SSD, gallery indices per probe row

    rng = np.random.default_rng(RNG_SEED)
    top10_diffs = np.empty(n)
    random_diffs = np.empty(n)

    all_gallery_idx = np.arange(n)
    for i in range(n):
        row_order = order[i]
        wrong_order = row_order[row_order != i]  # drop true identity wherever it ranked
        top10_idx = wrong_order[:10]
        top10_diffs[i] = np.mean(np.abs(mean_run_length[top10_idx] - mean_run_length[i]))

        # Random baseline: average over N_RANDOM_REPS draws of 10 random *other* subjects
        reps = np.empty(N_RANDOM_REPS)
        candidates = all_gallery_idx[all_gallery_idx != i]
        for r in range(N_RANDOM_REPS):
            rand_idx = rng.choice(candidates, size=10, replace=False)
            reps[r] = np.mean(np.abs(mean_run_length[rand_idx] - mean_run_length[i]))
        random_diffs[i] = reps.mean()

    print("\n--- 2. Confusability test (mean |run-length diff|, averaged over all probes) ---")
    print(f"  Top-10 DFT-ranked wrong candidates: {top10_diffs.mean():.4f} (+/- {top10_diffs.std():.4f} across probes)")
    print(f"  10 random gallery subjects ({N_RANDOM_REPS} reps averaged): {random_diffs.mean():.4f} "
          f"(+/- {random_diffs.std():.4f} across probes)")
    ratio = top10_diffs.mean() / random_diffs.mean()
    print(f"  Ratio (top10 / random): {ratio:.4f}  (<< 1 would indicate DFT is reading run length)")

    # ================================================================
    # 3. Failure clustering
    # ================================================================
    ranks_npz = np.load("stage2_ranks.npz")
    ranks_dft = ranks_npz["ranks_DFT"]
    assert len(ranks_dft) == n, "stage2_ranks.npz subject count mismatch with current gallery load"
    failed_idx = np.where(ranks_dft > FAILURE_RANK_THRESHOLD)[0]
    print(f"\n--- 3. Failure clustering ({len(failed_idx)} failed probes, rank > {FAILURE_RANK_THRESHOLD}) ---")
    print(f"  Failed-subject run length: mean={mean_run_length[failed_idx].mean():.3f}  "
          f"std={mean_run_length[failed_idx].std():.3f}")
    print(f"  Population run length:     mean={mean_run_length.mean():.3f}  std={mean_run_length.std():.3f}")

    # Local density (Gaussian KDE) of the run-length distribution, evaluated at each
    # subject's own run length -- higher = more mutually-confusable neighbors nearby.
    kde = gaussian_kde(mean_run_length)
    density_at_subject = kde(mean_run_length)
    rng2 = np.random.default_rng(RNG_SEED + 1)
    random_sample_idx = rng2.choice(n, size=len(failed_idx), replace=False)
    density_failed = density_at_subject[failed_idx]
    density_random = density_at_subject[random_sample_idx]
    density_all = density_at_subject
    print(f"  Local run-length density at failed subjects:        mean={density_failed.mean():.4f}")
    print(f"  Local run-length density at {len(failed_idx)} random subjects: mean={density_random.mean():.4f}")
    print(f"  Local run-length density averaged over all {n} subjects:    mean={density_all.mean():.4f}")
    print(f"  Ratio (failed / population): {density_failed.mean() / density_all.mean():.4f}  "
          f"(>> 1 would indicate failures cluster in dense/confusable regions)")

    # ================================================================
    # Plots
    # ================================================================
    fig, (ax_corr, ax_hist) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax_corr.scatter(mean_run_length, peak_bin, s=8, alpha=0.25, color=COLOR_ALL, edgecolors="none")
    x_ref = np.linspace(mean_run_length.min(), mean_run_length.max(), 200)
    ax_corr.plot(x_ref, N_COLS / (2 * x_ref), color="black", linestyle="--", linewidth=1.5,
                 label=r"theoretical: peak $\approx$ 512 / (2 $\times$ run length)")
    ax_corr.set_xlabel("Mean run length (IC1, circular, avg over 32 rows)", size=12)
    ax_corr.set_ylabel("DFT peak bin (excl. DC)", size=12)
    ax_corr.set_title(f"Mean run length vs DFT peak bin\nPearson r={pearson_r:.3f}, Spearman rho={spearman_rho:.3f}", size=12)
    ax_corr.legend(fontsize=9)
    ax_corr.grid(True, alpha=0.3)

    bins_hist = np.linspace(mean_run_length.min(), mean_run_length.max(), 50)
    ax_hist.hist(mean_run_length, bins=bins_hist, density=True, alpha=0.75,
                 color="white", edgecolor=COLOR_ALL, linewidth=1.5,
                 label=f"All {n} subjects  μ={mean_run_length.mean():.2f}")
    ax_hist.hist(mean_run_length[failed_idx], bins=bins_hist, density=True, alpha=0.75,
                 color="white", edgecolor=COLOR_FAIL, linewidth=1.5,
                 label=f"DFT failures (n={len(failed_idx)})  μ={mean_run_length[failed_idx].mean():.2f}")
    ax_hist.set_xlabel("Mean run length (IC1)", size=12)
    ax_hist.set_ylabel("Probability density", size=12)
    ax_hist.set_title("Run-length distribution: all subjects vs DFT failures", size=12)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    fig.suptitle("Negative control: is DFT reading SIC-Gen's per-subject run length?", size=14)
    plt.tight_layout()
    out_path = Path("stage2_run_length_artifact_check.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")

    np.savez(
        "stage2_run_length_artifact_check.npz",
        mean_run_length=mean_run_length,
        peak_bin=peak_bin,
        peak_mag=peak_mag,
        top10_diffs=top10_diffs,
        random_diffs=random_diffs,
        failed_idx=failed_idx,
        density_at_subject=density_at_subject,
    )
    print("Saved arrays to stage2_run_length_artifact_check.npz")


if __name__ == "__main__":
    main()
