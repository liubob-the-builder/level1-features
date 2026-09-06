#!/usr/bin/env python3
"""Bit-level Hamming-distance validation of SIC-Gen codes against real CASIA codes.

Closes the gap left by experiments/validate_gen.py, which measured genuine/impostor
fractional-HD distributions for SIC-Gen only, on a 100-subject generated directory,
and saved no numeric result file (its output survives only as the legend of
results/hd2.png). This script measures the *same* quantities on

  - the full 2000-subject SIC-Gen evaluation gallery
    (sic-gen/sicgen_gallery_2000subjects/), and
  - the full real CASIA-Iris-Thousand export
    (casia-extraction/casia-codes-2d/, 1998 usable identities),

side by side, so the claim "SIC-Gen reproduces real iris codes' bit-level
statistics" is backed by a measurement made here rather than by the figures
reported in the SIC-Gen paper.

Reported per dataset:
  - genuine fractional HD, +/-max_shift aligned
  - impostor fractional HD, unaligned and +/-max_shift aligned
  - degrees of freedom from the *unaligned* impostor distribution, using
    SIC-Gen's own estimator mu(1-mu)/sigma^2 (sic-gen/validation.py's
    degrees_of_freedom), which is the distribution that estimator is meant for
  - d' between the aligned-genuine and unaligned-impostor distributions
  - mask valid-bit fraction
  - run-length distribution (mean/std run length + histogram), which is the
    third statistic the write-up claims SIC-Gen reproduces and which was
    previously measured on the synthetic side only

ROTATION GEOMETRY (the one place the two datasets must be treated differently)
-----------------------------------------------------------------------------
Both datasets store 32x512 codes, but the 512 columns do not mean the same thing:

  SIC-Gen : 512 genuine angular positions. A rotation is np.roll over the whole
            width -- this is what SIC-Gen's own Template.hamming_distance and
            Template.rotate do, so it is used unchanged here.

  Real    : Open Iris returns 2 Gabor scales of shape (16, 256, 2); extract_casia.py
            concatenates the 2 phase channels along the angular axis to (16, 512)
            and vstacks the scales to (32, 512). Columns 0-255 are channel 0 at
            angles 0-255 and columns 256-511 are channel 1 at the *same* 256
            angles. A rotation by k angular positions is therefore a roll of each
            256-column half by k, NOT a roll of the full 512 -- rolling the full
            width wraps channel 1's trailing columns into channel 0 and mixes the
            two phase channels at the seam.

Rolling the full width is only mildly wrong (it corrupts 2k of 512 columns), so
this script measures both conventions on the real data and records them, rather
than just asserting the choice; see "rotation_convention_check" in the output.

Method: fractional HD for all pairs at once, via two matrix products rather than a
Python loop over pairs. With bipolar bits x = 2t-1 and mask m, writing u = m*x:

    u_a . u_b = (#agreeing valid bits) - (#disagreeing valid bits)
    m_a . m_b = n_valid
    => fractional HD = (n_valid - u_a . u_b) / (2 * n_valid)

Both dot products are exact in float32 here (integer magnitudes <= 16384, well
inside float32's 24-bit mantissa), so this is not an approximation.

Usage:
    python level1-features/experiments/hd_distribution_real_vs_synthetic.py
    python level1-features/experiments/hd_distribution_real_vs_synthetic.py --quick
    python level1-features/experiments/hd_distribution_real_vs_synthetic.py --max-shift 7
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_SICGEN = _ROOT / "sic-gen"
_SYNTH_GALLERY = _SICGEN / "sicgen_gallery_2000subjects"
_SYNTH_METADATA = _SYNTH_GALLERY / "metadata.json"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_CASIA_MANIFEST = _CASIA_DIR / "manifest.json"
_RESULTS = _LEVEL1 / "results"
_OUT_JSON = _RESULTS / "hd_distribution_real_vs_synthetic.json"
_OUT_PNG = _RESULTS / "hd_distribution_real_vs_synthetic.png"

DEFAULT_MAX_SHIFT = 7          # matches validate_gen.py's default and FilterFHEConfig.max_shift
N_ROWS, N_COLS = 32, 512
PROBE_STEMS = [str(i) for i in range(2, 11)]
EXPECTED_N_GALLERY = 1998
RUNLENGTH_HIST_CAP = 20        # run lengths >= this are pooled into the final bin
GEMM_DTYPE = np.float32


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_bit_file(path: Path) -> np.ndarray:
    """Load a 32x512 '0'/'1' text file (the format shared by both datasets)."""
    with open(path) as f:
        return np.array([list(r.strip()) for r in f if r.strip()], dtype=np.uint8)


def load_pair(directory: Path, stem: str):
    return (load_bit_file(directory / f"{stem}_template.txt"),
            load_bit_file(directory / f"{stem}_mask.txt"))


def get_git_commit(repo_dir: Path) -> str:
    """Read-only lookup of the current commit, matching the other experiment scripts."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def rotate(arr: np.ndarray, shift: int, n_blocks: int) -> np.ndarray:
    """Circularly shift along the angular axis.

    n_blocks=1 rolls the full width (SIC-Gen's convention: 512 angular positions).
    n_blocks=2 rolls each half independently (real codes: 2 phase channels over
    the same 256 angles, so a rotation moves each channel by the same k).
    """
    if shift == 0:
        return arr
    if n_blocks == 1:
        return np.roll(arr, shift, axis=-1)
    width = arr.shape[-1] // n_blocks
    return np.concatenate(
        [np.roll(arr[..., i * width:(i + 1) * width], shift, axis=-1)
         for i in range(n_blocks)], axis=-1)


# ---------------------------------------------------------------------------
# Fractional Hamming distance
# ---------------------------------------------------------------------------

def to_bipolar(templates: np.ndarray, masks: np.ndarray):
    """Flatten to (n, n_bits) and return (u = m*(2t-1), m) as GEMM_DTYPE."""
    n = templates.shape[0]
    t = templates.reshape(n, -1).astype(GEMM_DTYPE)
    m = masks.reshape(n, -1).astype(GEMM_DTYPE)
    return m * (2.0 * t - 1.0), m


def hd_matrix(u_a, m_a, u_b, m_b) -> np.ndarray:
    """All-pairs fractional HD between set A and set B. Returns (n_a, n_b)."""
    n_valid = m_a @ m_b.T
    agree_minus_disagree = u_a @ u_b.T
    with np.errstate(divide="ignore", invalid="ignore"):
        hd = (n_valid - agree_minus_disagree) / (2.0 * n_valid)
    return np.where(n_valid > 0, hd, 1.0)


def hd_rowwise(u_a, m_a, u_b, m_b) -> np.ndarray:
    """Fractional HD for the i-th row of A against the i-th row of B."""
    n_valid = np.einsum("ij,ij->i", m_a, m_b)
    agree_minus_disagree = np.einsum("ij,ij->i", u_a, u_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        hd = (n_valid - agree_minus_disagree) / (2.0 * n_valid)
    return np.where(n_valid > 0, hd, 1.0)


def genuine_hds(probe_t, probe_m, gal_t, gal_m, max_shift, n_blocks, chunk=2048):
    """Best-of-shifts HD for each (probe_i, gallery_i) pair, in chunks."""
    out = np.empty(probe_t.shape[0], dtype=np.float64)
    for lo in range(0, probe_t.shape[0], chunk):
        hi = min(lo + chunk, probe_t.shape[0])
        u_g, m_g = to_bipolar(gal_t[lo:hi], gal_m[lo:hi])
        best = np.full(hi - lo, np.inf)
        for s in range(-max_shift, max_shift + 1):
            u_p, m_p = to_bipolar(rotate(probe_t[lo:hi], s, n_blocks),
                                  rotate(probe_m[lo:hi], s, n_blocks))
            np.minimum(best, hd_rowwise(u_g, m_g, u_p, m_p), out=best)
        out[lo:hi] = best
    return out


def impostor_hds(templates, masks, max_shift, n_blocks):
    """Upper-triangle impostor HDs, unaligned and best-of-shifts aligned.

    All pairs are used -- no sampling -- since the matrix form makes the full
    set affordable (1998^2 and 2000^2 pairs).
    """
    u0, m0 = to_bipolar(templates, masks)
    iu = np.triu_indices(templates.shape[0], k=1)

    unaligned = hd_matrix(u0, m0, u0, m0)[iu].astype(np.float64)

    best = None
    for s in range(-max_shift, max_shift + 1):
        u_s, m_s = to_bipolar(rotate(templates, s, n_blocks), rotate(masks, s, n_blocks))
        hd_s = hd_matrix(u0, m0, u_s, m_s)
        # Pair (i,j) is compared at shift s and, on the transpose, at shift -s,
        # so the elementwise min of the matrix with its transpose covers both
        # directions before the triangle is taken.
        np.minimum(hd_s, hd_s.T, out=hd_s)
        best = hd_s if best is None else np.minimum(best, hd_s)
    aligned = best[iu].astype(np.float64)
    return unaligned, aligned


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def degrees_of_freedom(mu: float, sigma: float) -> float:
    """SIC-Gen's estimator (sic-gen/validation.py), kept unrounded here."""
    return float(mu * (1.0 - mu) / (sigma * sigma)) if sigma > 0 else float("nan")


def d_prime(genuine: np.ndarray, impostor: np.ndarray) -> float:
    return float(abs(impostor.mean() - genuine.mean())
                 / np.sqrt(0.5 * (genuine.var() + impostor.var())))


def describe(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a))}


def run_length_stats(templates: np.ndarray, masks: np.ndarray, n_blocks: int) -> dict:
    """Run lengths within maximal contiguous valid-bit segments.

    Matches features/rl_features.py's mask-aware convention: an occluded bit
    terminates the run it interrupts and is not itself counted, and no run
    bridges an occluded gap. Segments are non-circular, applied identically to
    both datasets so the comparison is like-for-like. Each 256-column phase
    channel of a real code is walked separately (n_blocks=2) so no run is
    counted across the channel seam.
    """
    counts = np.zeros(RUNLENGTH_HIST_CAP, dtype=np.int64)
    total_len, total_runs = 0, 0
    width = templates.shape[-1] // n_blocks

    for t, m in zip(templates, masks):
        for row_t, row_m in zip(t, m):
            for b in range(n_blocks):
                bits = row_t[b * width:(b + 1) * width]
                valid = row_m[b * width:(b + 1) * width].astype(bool)
                if not valid.any():
                    continue
                # Split into maximal valid segments, then into runs within each.
                seg_edges = np.flatnonzero(np.diff(valid.astype(np.int8))) + 1
                for start, end in zip(np.concatenate(([0], seg_edges)),
                                      np.concatenate((seg_edges, [valid.size]))):
                    if not valid[start] or end - start < 1:
                        continue
                    seg = bits[start:end]
                    run_edges = np.flatnonzero(np.diff(seg.astype(np.int8))) + 1
                    lengths = np.diff(np.concatenate(([0], run_edges, [seg.size])))
                    counts += np.bincount(
                        np.minimum(lengths, RUNLENGTH_HIST_CAP) - 1,
                        minlength=RUNLENGTH_HIST_CAP)[:RUNLENGTH_HIST_CAP]
                    total_len += int(lengths.sum())
                    total_runs += int(lengths.size)

    hist = counts / counts.sum() if counts.sum() else counts.astype(float)
    # Mean over the capped histogram understates the tail; report the exact mean
    # from the uncapped totals alongside it.
    return {
        "mean_run_length": total_len / total_runs if total_runs else float("nan"),
        "n_runs": total_runs,
        "histogram_bin_labels": [str(i) for i in range(1, RUNLENGTH_HIST_CAP)]
                                + [f">={RUNLENGTH_HIST_CAP}"],
        "histogram_normalised": [float(x) for x in hist],
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_synthetic(limit=None):
    dirs = sorted((d for d in _SYNTH_GALLERY.iterdir() if d.is_dir()), key=lambda p: int(p.name))
    if limit:
        dirs = dirs[:limit]
    ic1_t, ic1_m, ic2_t, ic2_m = [], [], [], []
    for d in dirs:
        t1, m1 = load_pair(d, "1")
        t2, m2 = load_pair(d, "2")
        ic1_t.append(t1); ic1_m.append(m1); ic2_t.append(t2); ic2_m.append(m2)
    return (np.array(ic1_t), np.array(ic1_m), np.array(ic2_t), np.array(ic2_m),
            [d.name for d in dirs])


def load_real(limit=None):
    """Gallery = stem '1' of each gallery_ok identity; probes = stems '2'..'10'."""
    manifest = json.loads(_CASIA_MANIFEST.read_text())
    idents = [i for i in manifest["identities"] if i["gallery_ok"]]
    if limit:
        idents = idents[:limit]

    gal_t, gal_m, names = [], [], []
    probe_t, probe_m, probe_gal_t, probe_gal_m = [], [], [], []
    for ident in idents:
        d = _CASIA_2D_DIR / ident["identity"]
        t1, m1 = load_pair(d, "1")
        gal_t.append(t1); gal_m.append(m1); names.append(ident["identity"])
        for stem in PROBE_STEMS:
            if not (d / f"{stem}_template.txt").exists():
                continue
            tp, mp = load_pair(d, stem)
            probe_t.append(tp); probe_m.append(mp)
            probe_gal_t.append(t1); probe_gal_m.append(m1)
    return (np.array(gal_t), np.array(gal_m), np.array(probe_t), np.array(probe_m),
            np.array(probe_gal_t), np.array(probe_gal_m), names)


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def analyse(label, gal_t, gal_m, probe_t, probe_m, probe_gal_t, probe_gal_m,
            max_shift, n_blocks):
    print(f"\n[{label}] {gal_t.shape[0]} gallery codes, {probe_t.shape[0]} genuine pairs "
          f"(rotation: {'full-width roll' if n_blocks == 1 else 'per-channel roll'}, "
          f"+/-{max_shift})")

    print(f"[{label}]   genuine ...", flush=True)
    gen = genuine_hds(probe_t, probe_m, probe_gal_t, probe_gal_m, max_shift, n_blocks)

    print(f"[{label}]   impostor ({gal_t.shape[0] * (gal_t.shape[0] - 1) // 2} pairs) ...",
          flush=True)
    imp_un, imp_al = impostor_hds(gal_t, gal_m, max_shift, n_blocks)

    print(f"[{label}]   run lengths ...", flush=True)
    rl = run_length_stats(gal_t, gal_m, n_blocks)

    g, iu_, ia = describe(gen), describe(imp_un), describe(imp_al)
    valid_frac = float(gal_m.reshape(gal_m.shape[0], -1).mean())

    result = {
        "n_gallery_codes": int(gal_t.shape[0]),
        "n_genuine_pairs": int(gen.size),
        "n_impostor_pairs": int(imp_un.size),
        "impostor_pair_sampling": "none -- all gallery pairs used exhaustively",
        "rotation_convention": ("full-width roll over 512 angular positions"
                                if n_blocks == 1 else
                                "per-channel roll: each 256-column phase channel "
                                "rolled by the same k"),
        "max_shift": max_shift,
        "genuine_aligned": g,
        "impostor_unaligned": iu_,
        "impostor_aligned": ia,
        "degrees_of_freedom_impostor_unaligned": degrees_of_freedom(iu_["mean"], iu_["std"]),
        "d_prime_genuine_aligned_vs_impostor_unaligned": d_prime(gen, imp_un),
        "mask_valid_fraction_mean": valid_frac,
        "run_lengths": rl,
    }

    print(f"[{label}]   genuine  aligned  : mu={g['mean']:.4f} sd={g['std']:.4f}")
    print(f"[{label}]   impostor unaligned: mu={iu_['mean']:.4f} sd={iu_['std']:.4f}  "
          f"DoF={result['degrees_of_freedom_impostor_unaligned']:.1f}")
    print(f"[{label}]   impostor aligned  : mu={ia['mean']:.4f} sd={ia['std']:.4f}")
    print(f"[{label}]   d'={result['d_prime_genuine_aligned_vs_impostor_unaligned']:.2f}  "
          f"valid={valid_frac:.3f}  mean run length={rl['mean_run_length']:.3f}")
    return result, gen, imp_un, imp_al


def rotation_convention_check(probe_t, probe_m, probe_gal_t, probe_gal_m, max_shift, n=2000):
    """Record both rotation conventions on real genuine pairs, so the choice is auditable."""
    n = min(n, probe_t.shape[0])
    out = {}
    for label, n_blocks in (("per_channel_roll", 2), ("full_width_roll", 1)):
        hds = genuine_hds(probe_t[:n], probe_m[:n], probe_gal_t[:n], probe_gal_m[:n],
                          max_shift, n_blocks)
        out[label] = {"mean_genuine_hd": float(hds.mean()), "n_pairs": int(n)}
    out["note"] = (
        "Lower is better: the correct geometry recovers more of the true rotation. "
        "per_channel_roll is used for the real dataset. The two differ only at the "
        "channel seam (2k of 512 columns), so the gap is small; it is recorded to "
        "show the choice is not load-bearing.")
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(panels, out_path, max_shift):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 9))
    bins = np.linspace(0, 1, 61)

    for row, (label, gen, imp_un, imp_al, res) in enumerate(panels):
        for col, (imp, title) in enumerate((
            (imp_al, f"Aligned (genuine +/-{max_shift}, impostor +/-{max_shift})"),
            (imp_un, f"Unaligned (genuine +/-{max_shift}, impostor no shifts)"),
        )):
            ax = axes[row][col]
            ax.hist(gen, bins=bins, density=True, color="white", edgecolor="green",
                    linewidth=1.5, label=f"Genuine  mu={gen.mean():.3f} sd={gen.std():.3f}")
            ax.hist(imp, bins=bins, density=True, color="white", edgecolor="red",
                    linewidth=1.5, label=f"Impostor mu={imp.mean():.3f} sd={imp.std():.3f}")
            ax.set_xlabel("Fractional HD")
            ax.set_ylabel("Probability density")
            ax.set_title(f"{label} -- {title}", size=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        ax = axes[row][2]
        rl = res["run_lengths"]
        x = np.arange(1, RUNLENGTH_HIST_CAP + 1)
        ax.bar(x, rl["histogram_normalised"], color="white", edgecolor="steelblue",
               linewidth=1.5)
        ax.set_xlabel(f"Run length (bits; final bin = >={RUNLENGTH_HIST_CAP})")
        ax.set_ylabel("Fraction of runs")
        ax.set_title(f"{label} -- run lengths (mean {rl['mean_run_length']:.2f})", size=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bit-level statistics: SIC-Gen vs real CASIA-Iris-Thousand", size=14)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-shift", type=int, default=DEFAULT_MAX_SHIFT,
                    help=f"max circular shift for alignment (default: {DEFAULT_MAX_SHIFT})")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on the first 100 subjects/identities of each dataset; "
                         "writes to *_quick.json/png so full results are never overwritten")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    limit = 100 if args.quick else None
    out_json = _OUT_JSON.with_name(_OUT_JSON.stem + "_quick.json") if args.quick else _OUT_JSON
    out_png = _OUT_PNG.with_name(_OUT_PNG.stem + "_quick.png") if args.quick else _OUT_PNG
    if out_json.exists():
        sys.exit(f"Refusing to overwrite existing {out_json} -- move it aside first.")

    print(f"Loading synthetic gallery from {_SYNTH_GALLERY} ...", flush=True)
    s_ic1_t, s_ic1_m, s_ic2_t, s_ic2_m, s_names = load_synthetic(limit)
    print(f"Loading real codes from {_CASIA_2D_DIR} ...", flush=True)
    r_gal_t, r_gal_m, r_pr_t, r_pr_m, r_prgal_t, r_prgal_m, r_names = load_real(limit)

    if not args.quick and r_gal_t.shape[0] != EXPECTED_N_GALLERY:
        sys.exit(f"Expected {EXPECTED_N_GALLERY} usable real galleries, "
                 f"got {r_gal_t.shape[0]} -- check the manifest.")

    synth_res, s_gen, s_imp_un, s_imp_al = analyse(
        "synthetic", s_ic1_t, s_ic1_m, s_ic2_t, s_ic2_m, s_ic1_t, s_ic1_m,
        args.max_shift, n_blocks=1)
    real_res, r_gen, r_imp_un, r_imp_al = analyse(
        "real", r_gal_t, r_gal_m, r_pr_t, r_pr_m, r_prgal_t, r_prgal_m,
        args.max_shift, n_blocks=2)

    print("\n[real] rotation-convention check ...", flush=True)
    rot_check = rotation_convention_check(r_pr_t, r_pr_m, r_prgal_t, r_prgal_m, args.max_shift)
    for k, v in rot_check.items():
        if isinstance(v, dict):
            print(f"  {k}: mean genuine HD = {v['mean_genuine_hd']:.4f} over {v['n_pairs']} pairs")

    synth_meta = json.loads(_SYNTH_METADATA.read_text())
    casia_meta = json.loads(_CASIA_MANIFEST.read_text())

    out = {
        "metadata": {
            "purpose": "Measure genuine/impostor fractional-HD distributions, degrees of "
                       "freedom, mask occupancy and run-length distributions for SIC-Gen and "
                       "real CASIA codes under one method, so real-vs-synthetic bit-level "
                       "agreement is measured here rather than cited from the SIC-Gen paper.",
            "supersedes": "experiments/validate_gen.py / results/hd2.png -- synthetic only, "
                          "100-subject generated directory, no numeric output saved.",
            "date": date.today().isoformat(),
            "git_commit": get_git_commit(_LEVEL1),
            "max_shift": args.max_shift,
            "quick_mode": bool(args.quick),
            "subject_limit_per_dataset": limit,
            "synthetic": {
                "gallery_path": str(_SYNTH_GALLERY.relative_to(_ROOT)),
                "subject_count": int(s_ic1_t.shape[0]),
                "seed": synth_meta.get("seed"),
                "num_processes": synth_meta.get("num_processes"),
                "generator_git_commit": synth_meta.get("git_commit"),
                "pairs": "gallery = IC1 (1_template.txt); genuine probe = IC2 (2_template.txt); "
                         "impostor pairs = all IC1 pairs",
            },
            "real": {
                "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
                "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
                "open_iris_version": casia_meta.get("open_iris_version"),
                "identity_count_usable": int(r_gal_t.shape[0]),
                "subject_selection_rule": casia_meta.get("subject_selection_rule"),
                "seeds_used": None,
                "excluded_identities": ["524_L", "668_L"],
                "exclusion_reason": "gallery image (index 00) failed segmentation "
                                    "(gallery_ok=False in manifest.json).",
                "pairs": "gallery = stem '1' (image index 00); genuine probes = stems '2'-'10'; "
                         "impostor pairs = all gallery pairs",
            },
            "mask_convention": "1 = valid in both datasets (Open Iris native; SIC-Gen's "
                               "Template.add_noise stores the negation of the occlusion mask).",
            "degrees_of_freedom_estimator": "mu(1-mu)/sigma^2 applied to the UNALIGNED impostor "
                                            "distribution, matching sic-gen/validation.py.",
            "method_note": "All-pairs fractional HD computed via two matrix products on bipolar "
                           "masked bits (exact in float32 at this bit count); no pair sampling.",
            "run_length_note": "Run lengths counted within maximal contiguous valid-bit segments "
                               "(features/rl_features.py's mask-aware convention), non-circular, "
                               "applied identically to both datasets. Real codes are walked one "
                               "256-column phase channel at a time so no run crosses the seam.",
        },
        "results": {"synthetic": synth_res, "real": real_res},
        "rotation_convention_check_real": rot_check,
        "comparison": {
            "impostor_unaligned_mean": {
                "synthetic": synth_res["impostor_unaligned"]["mean"],
                "real": real_res["impostor_unaligned"]["mean"],
                "difference": synth_res["impostor_unaligned"]["mean"]
                              - real_res["impostor_unaligned"]["mean"],
            },
            "degrees_of_freedom": {
                "synthetic": synth_res["degrees_of_freedom_impostor_unaligned"],
                "real": real_res["degrees_of_freedom_impostor_unaligned"],
                "ratio_synthetic_over_real":
                    synth_res["degrees_of_freedom_impostor_unaligned"]
                    / real_res["degrees_of_freedom_impostor_unaligned"]
                    if real_res["degrees_of_freedom_impostor_unaligned"] else None,
            },
            "genuine_aligned_mean": {
                "synthetic": synth_res["genuine_aligned"]["mean"],
                "real": real_res["genuine_aligned"]["mean"],
            },
            "mean_run_length": {
                "synthetic": synth_res["run_lengths"]["mean_run_length"],
                "real": real_res["run_lengths"]["mean_run_length"],
            },
            "mask_valid_fraction_mean": {
                "synthetic": synth_res["mask_valid_fraction_mean"],
                "real": real_res["mask_valid_fraction_mean"],
            },
        },
    }

    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_json}")

    if not args.no_plot:
        make_plot([("SIC-Gen", s_gen, s_imp_un, s_imp_al, synth_res),
                   ("Real CASIA", r_gen, r_imp_un, r_imp_al, real_res)],
                  out_png, args.max_shift)


if __name__ == "__main__":
    main()
