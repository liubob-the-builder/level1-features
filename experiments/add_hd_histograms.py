#!/usr/bin/env python3
"""Add binned HD histograms to results/hd_distribution_real_vs_synthetic.json, in place.

hd_distribution_real_vs_synthetic.py reports each fractional-HD distribution only as
summary statistics (n / mean / std / min / max / median) and discards the underlying
values. A plotting script therefore has to fall back on drawing normal densities from
the reported mean and standard deviation, which misrepresents any non-Gaussian shape --
notably the secondary mode near HD 0.33-0.37 in the SIC-Gen genuine distribution.

This script recomputes the six distributions with the original code, imported
unmodified, and merges a binned density into each one:

    results.{synthetic,real}.{genuine_aligned,impostor_unaligned,impostor_aligned}
        + bin_edges           list[float], length HD_BINS + 1, ascending
        + histogram_density   list[float], length HD_BINS, density-normalised
        + fraction_above_range  float, mass discarded above HD_RANGE[1]

All six share the same bin edges. density=True is required: the genuine sets have
2000 / 16457 samples against the impostor sets' ~2e6, so raw counts are not comparable
on one axis.

The values are recomputed rather than re-derived because they were never saved. That
recomputation is deterministic (no sampling anywhere in the original method), so this
script asserts every recomputed summary statistic matches the stored one before it
writes; a mismatch means the histograms would not describe the file they are being
added to, and the script aborts instead.

Only the analysis of the whole 32x512 code is covered here -- no Gabor-scale split; see
audit_singlescale_regions.py for the per-scale slices.

Nothing is removed or renamed. Existing keys, key order, `run_lengths`, the
degrees-of-freedom and mask-occupancy fields and `metadata` are all preserved as-is;
`metadata.hd_histogram` is added to document the binning.

Usage:
    python level1-features/experiments/add_hd_histograms.py
    python level1-features/experiments/add_hd_histograms.py --dry-run
    python level1-features/experiments/add_hd_histograms.py --force   # re-bin if present
"""

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_RESULTS = _LEVEL1 / "results"
_TARGET = _RESULTS / "hd_distribution_real_vs_synthetic.json"

sys.path.insert(0, str(_EXPERIMENTS))

from hd_distribution_real_vs_synthetic import (  # noqa: E402
    DEFAULT_MAX_SHIFT,
    describe,
    genuine_hds,
    get_git_commit,
    impostor_hds,
    load_real,
    load_synthetic,
)

HD_BINS = 130                 # 0.005 wide over the plotted range
HD_RANGE = (0.0, 0.65)        # matches the figure's x-axis truncation

# Summary fields that must agree with the stored file for the histograms to be valid.
CHECKED_FIELDS = ("mean", "std", "min", "max", "median")
TOLERANCE = 1e-6              # float32 GEMM reduction order can differ run to run


def hd_histogram(hd_values: np.ndarray) -> dict:
    """Density-normalised histogram on the shared bin edges, plus truncated mass."""
    a = np.asarray(hd_values, dtype=float)
    counts, edges = np.histogram(a, bins=HD_BINS, range=HD_RANGE, density=True)
    return {
        "bin_edges": edges.tolist(),
        "histogram_density": counts.tolist(),
        "fraction_above_range": float((a > HD_RANGE[1]).mean()),
    }


def check_against_stored(label: str, values: np.ndarray, stored: dict) -> list:
    """Return a list of human-readable mismatches between recomputed and stored stats."""
    fresh = describe(values)
    problems = []
    if fresh["n"] != stored["n"]:
        problems.append(f"{label}: n {fresh['n']} != stored {stored['n']}")
    for f in CHECKED_FIELDS:
        if abs(fresh[f] - stored[f]) > TOLERANCE:
            problems.append(f"{label}: {f} {fresh[f]!r} != stored {stored[f]!r}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="recompute and verify, print what would be added, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-bin even if histogram keys are already present")
    args = ap.parse_args()

    if not _TARGET.exists():
        sys.exit(f"{_TARGET} not found -- run hd_distribution_real_vs_synthetic.py first.")

    doc = json.loads(_TARGET.read_text())
    max_shift = doc["metadata"]["max_shift"]
    if doc["metadata"].get("quick_mode"):
        sys.exit("Target file was produced in --quick mode; refusing to bin it.")

    targets = [(ds, dist) for ds in ("synthetic", "real")
               for dist in ("genuine_aligned", "impostor_unaligned", "impostor_aligned")]
    already = [f"{ds}.{dist}" for ds, dist in targets
               if "bin_edges" in doc["results"][ds][dist]]
    if already and not args.force:
        sys.exit(f"Histogram keys already present on: {', '.join(already)}. "
                 f"Re-run with --force to recompute them.")

    print(f"Recomputing the six distributions at max_shift={max_shift} "
          f"(deterministic; verified against the stored summaries) ...\n", flush=True)

    print("Loading synthetic gallery ...", flush=True)
    s_ic1_t, s_ic1_m, s_ic2_t, s_ic2_m, _ = load_synthetic(None)
    print("Loading real codes ...", flush=True)
    r_gal_t, r_gal_m, r_pr_t, r_pr_m, r_prgal_t, r_prgal_m, _ = load_real(None)

    values = {}
    for label, (gal_t, gal_m, pr_t, pr_m, prgal_t, prgal_m, n_blocks) in (
        ("synthetic", (s_ic1_t, s_ic1_m, s_ic2_t, s_ic2_m, s_ic1_t, s_ic1_m, 1)),
        ("real", (r_gal_t, r_gal_m, r_pr_t, r_pr_m, r_prgal_t, r_prgal_m, 2)),
    ):
        print(f"[{label}] genuine ...", flush=True)
        gen = genuine_hds(pr_t, pr_m, prgal_t, prgal_m, max_shift, n_blocks)
        print(f"[{label}] impostor (unaligned + aligned over "
              f"{2 * max_shift + 1} shifts) ...", flush=True)
        imp_un, imp_al = impostor_hds(gal_t, gal_m, max_shift, n_blocks)
        values[(label, "genuine_aligned")] = gen
        values[(label, "impostor_unaligned")] = imp_un
        values[(label, "impostor_aligned")] = imp_al

    print("\nVerifying recomputed values against the stored summaries ...")
    problems = []
    for ds, dist in targets:
        problems += check_against_stored(f"{ds}.{dist}", values[(ds, dist)],
                                         doc["results"][ds][dist])
    if problems:
        print("\nMISMATCH -- the recomputed distributions do not match the stored file:")
        for p in problems:
            print(f"  {p}")
        sys.exit("Refusing to write histograms that would not describe the stored statistics.")
    print("  all six match stored n/mean/std/min/max/median "
          f"(tolerance {TOLERANCE:g}).")

    print("\nBinning ...")
    for ds, dist in targets:
        h = hd_histogram(values[(ds, dist)])
        doc["results"][ds][dist].update(h)
        above = h["fraction_above_range"]
        print(f"  {ds:10s} {dist:20s} peak density={max(h['histogram_density']):7.3f}  "
              f"above range={above:.3e}" + ("  <-- truncated mass" if above > 0 else ""))

    doc["metadata"]["hd_histogram"] = {
        "added_by": "experiments/add_hd_histograms.py",
        "date_added": date.today().isoformat(),
        "git_commit_at_add": get_git_commit(_LEVEL1),
        "bins": HD_BINS,
        "range": list(HD_RANGE),
        "bin_width": (HD_RANGE[1] - HD_RANGE[0]) / HD_BINS,
        "density": True,
        "shared_bin_edges": "All six distributions use identical bin_edges.",
        "note": "bin_edges/histogram_density/fraction_above_range were added to each of "
                "results.{synthetic,real}.{genuine_aligned,impostor_unaligned,"
                "impostor_aligned} so a plot can be drawn from the measured shape rather "
                "than from a normal density fitted to mean/std. Values are density-"
                "normalised (integrating to 1 over the retained range) because the genuine "
                "and impostor sets differ in sample count by ~100x. Recomputed with the "
                "original code at the same max_shift and verified to match the stored "
                "n/mean/std/min/max/median before writing.",
        "truncation_note": "np.histogram drops values outside range and density=True "
                           "normalises over the retained values only; fraction_above_range "
                           "records the discarded mass per distribution.",
        "not_split_by_gabor_scale": "These cover the whole 32x512 code. Per-Gabor-scale "
                                    "slices live in results/audit_singlescale_results.json.",
    }

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # Write via a temp file in the same directory, then atomically replace, so an
    # interrupted write cannot leave the results file truncated.
    tmp = _TARGET.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, _TARGET)
    print(f"\nUpdated {_TARGET}")


if __name__ == "__main__":
    main()
