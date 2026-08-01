#!/usr/bin/env python3
"""Validate generated SIC-Gen iris codes by computing genuine and impostor HD distributions.

Always computes two comparisons side by side:
  - Aligned:   genuine ±rotations, impostor ±rotations
  - Unaligned: genuine ±rotations, impostor no shifts

Usage:
    python validate_gen.py -d generated/
    python validate_gen.py -d generated/ -r 10 -p 4 --plot hd.png
"""

import argparse
import itertools
import sys
import numpy as np
from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path

# Anchor to repo layout regardless of working directory:
# this file is in level1-features/experiments/ ; `template` (sic-gen's Template
# class) lives in ../../sic-gen/ and was not split out as a feature module.
_SICGEN = Path(__file__).resolve().parent.parent.parent / "sic-gen"
sys.path.insert(0, str(_SICGEN))

from template import Template


def fractional_hd(t1: np.ndarray, m1: np.ndarray,
                  t2: np.ndarray, m2: np.ndarray) -> float:
    """Fractional HD = differing bits in valid region / valid bits.
    Valid region = mask1 AND mask2. Single comparison, no shifts.
    """
    valid = np.logical_and(m1, m2)
    n_valid = np.count_nonzero(valid)
    if n_valid == 0:
        return 1.0
    return np.count_nonzero(np.logical_and(np.logical_xor(t1, t2), valid)) / n_valid


def fractional_hd_aligned(t1: np.ndarray, m1: np.ndarray,
                           t2: np.ndarray, m2: np.ndarray,
                           rotations: int = 10) -> float:
    """Fractional HD with circular shift alignment.
    Tries shifts in [-rotations, +rotations] and returns the minimum HD.
    """
    best_hd = 1.0
    for shift in range(-rotations, rotations + 1):
        t2_r = np.roll(t2, shift, axis=1)
        m2_r = np.roll(m2, shift, axis=1)
        hd = fractional_hd(t1, m1, t2_r, m2_r)
        if hd < best_hd:
            best_hd = hd
    return best_hd


def load_subject(subject_dir: Path):
    ref = Template.from_file(subject_dir / "1_template.txt", subject_dir / "1_mask.txt")
    probe = Template.from_file(subject_dir / "2_template.txt", subject_dir / "2_mask.txt")
    return subject_dir.name, ref, probe


def _genuine_hd(subject, rotations):
    _, ref, probe = subject
    return fractional_hd_aligned(ref._template, ref._mask, probe._template, probe._mask, rotations)


def _impostor_hd_aligned(pair, rotations):
    (_, ref1, _), (_, ref2, _) = pair
    return fractional_hd_aligned(ref1._template, ref1._mask, ref2._template, ref2._mask, rotations)


def _impostor_hd_unaligned(pair):
    (_, ref1, _), (_, ref2, _) = pair
    return fractional_hd(ref1._template, ref1._mask, ref2._template, ref2._mask)


def print_stats(label: str, values: list) -> None:
    a = np.array(values)
    print(f"  {label}:  n={len(a):>6}  mean={np.mean(a):.4f}  std={np.std(a):.4f}  "
          f"min={np.min(a):.4f}  max={np.max(a):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute genuine and impostor HD distributions for generated iris codes."
    )
    parser.add_argument("-d", "--directory", type=Path, default=_SICGEN / "generated",
                        help="folder containing generated iris codes (default: sic-gen/generated)")
    parser.add_argument("-r", "--rotations", type=int, default=7,
                        help="max circular shift for alignment (default: 7, i.e. 15 shifts total)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of CPU processes (default: 1)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="save histogram to this path (e.g. hd.png); omit to show interactively")
    args = parser.parse_args()

    subject_dirs = sorted(d for d in args.directory.iterdir() if d.is_dir())
    if not subject_dirs:
        print(f"No subject directories found in {args.directory}")
        return

    print(f"Loading {len(subject_dirs)} subjects from '{args.directory}' ...")
    subjects = [load_subject(d) for d in subject_dirs]
    impostor_pairs = list(itertools.combinations(subjects, 2))

    print(f"Computing {len(subjects)} genuine and {len(impostor_pairs)} impostor HDs "
          f"(rotations=±{args.rotations}) ...")

    compute_genuine = partial(_genuine_hd, rotations=args.rotations)
    compute_impostor_aligned = partial(_impostor_hd_aligned, rotations=args.rotations)

    if args.processes > 1:
        with Pool(args.processes) as pool:
            genuine_hds         = pool.map(compute_genuine,          subjects)
            impostor_aligned    = pool.map(compute_impostor_aligned,  impostor_pairs)
            impostor_unaligned  = pool.map(_impostor_hd_unaligned,    impostor_pairs)
    else:
        genuine_hds         = list(map(compute_genuine,         subjects))
        impostor_aligned    = list(map(compute_impostor_aligned, impostor_pairs))
        impostor_unaligned  = list(map(_impostor_hd_unaligned,   impostor_pairs))

    print(f"\n{'─'*60}")
    print(f"  Aligned  (genuine ±{args.rotations}, impostor ±{args.rotations})")
    print(f"{'─'*60}")
    print_stats("Genuine ", genuine_hds)
    print_stats("Impostor", impostor_aligned)

    print(f"\n{'─'*60}")
    print(f"  Unaligned  (genuine ±{args.rotations}, impostor no shifts)")
    print(f"{'─'*60}")
    print_stats("Genuine ", genuine_hds)
    print_stats("Impostor", impostor_unaligned)

    from matplotlib import pyplot as plt

    bins = np.linspace(0, 1, 51)
    fig, (ax_aligned, ax_unaligned) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, imp_hds, title in (
        (ax_aligned,   impostor_aligned,   f"Aligned  (genuine ±{args.rotations}, impostor ±{args.rotations})"),
        (ax_unaligned, impostor_unaligned, f"Unaligned  (genuine ±{args.rotations}, impostor no shifts)"),
    ):
        ax.hist(genuine_hds, bins=bins, density=True, alpha=0.75,
                color="white", edgecolor="green", linewidth=1.5,
                label=f"Genuine  μ={np.mean(genuine_hds):.3f} σ={np.std(genuine_hds):.3f}")
        ax.hist(imp_hds, bins=bins, density=True, alpha=0.75,
                color="white", edgecolor="red", linewidth=1.5,
                label=f"Impostor μ={np.mean(imp_hds):.3f} σ={np.std(imp_hds):.3f}")
        ax.set_xlabel("Fractional HD", size=12)
        ax.set_ylabel("Probability Density", size=12)
        ax.set_title(title, size=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"HD Distributions — {len(subjects)} subjects", size=14)
    plt.tight_layout()

    if args.plot:
        plt.savefig(str(args.plot), bbox_inches="tight")
        print(f"\nPlot saved to {args.plot}")
    else:
        plt.show()


if __name__ == "__main__":
    set_start_method("spawn")
    main()
