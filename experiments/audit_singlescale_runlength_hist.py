#!/usr/bin/env python3
"""Run-length histograms for the coarse single-scale single-phase CASIA region vs SIC-Gen.

audit_singlescale_regions.py found that of all the real slices tested, the coarse Gabor
scale at one phase channel (rows 0-15, cols 0-255) sits closest to SIC-Gen on both
degrees of freedom (245.0 vs 229.0) and mean run length (7.007 vs 6.010). This script
pulls the run-length distribution for that region out as a self-contained, plot-ready
file next to SIC-Gen's, so the two shapes can be drawn against each other.

Regions covered (whole-code and per-Gabor-scale HD statistics stay in
results/audit_singlescale_results.json; nothing there is recomputed or changed):

  casia_scale0_real   rows 0-15, cols   0-255  -- coarse Gabor scale, REAL phase channel
  casia_scale0_imag   rows 0-15, cols 256-511  -- coarse Gabor scale, IMAGINARY channel
  sicgen_full         rows 0-31, cols   0-511  -- SIC-Gen, whole code

The imaginary channel is included because "one phase channel" is a choice, not a given:
cols 0-255 is the real part only because iris_encoder.py stacks [real>0, imag>0] in that
order. Measuring both shows whether the choice is load-bearing or arbitrary.

run_length_stats is imported unmodified from hd_distribution_real_vs_synthetic (runs
within maximal contiguous valid-bit segments, mask-aware, non-circular, one phase
channel at a time). Its histogram caps at >=20; the discarded tail is 0.8% of runs for
the CASIA region and 0.4% for SIC-Gen, so the capped bins carry essentially the whole
distribution. n_blocks is 1 for the 256-column CASIA regions (a single channel) and 1
for SIC-Gen (512 genuine angles, no seam).

Degrees of freedom and the unaligned impostor distribution are recomputed for the two
CASIA regions as well, so the "which channel is closer to SIC-Gen" question can be
answered on both statistics rather than run length alone.

SIC-Gen's run lengths are recomputed here and checked against the stored value in
results/hd_distribution_real_vs_synthetic.json before anything is written.

Usage:
    python level1-features/experiments/audit_singlescale_runlength_hist.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_EXPERIMENTS = Path(__file__).resolve().parent
_LEVEL1 = _EXPERIMENTS.parent
_ROOT = _LEVEL1.parent
_RESULTS = _LEVEL1 / "results"
_REFERENCE_JSON = _RESULTS / "hd_distribution_real_vs_synthetic.json"
_OUT_JSON = _RESULTS / "audit_singlescale_runlength_hist.json"

sys.path.insert(0, str(_EXPERIMENTS))

from hd_distribution_real_vs_synthetic import (  # noqa: E402
    degrees_of_freedom,
    describe,
    get_git_commit,
    hd_matrix,
    load_real,
    load_synthetic,
    run_length_stats,
    to_bipolar,
    _CASIA_2D_DIR,
    _CASIA_MANIFEST,
    _SYNTH_GALLERY,
)


def enrich(rl: dict) -> dict:
    """Add plot-ready derived fields to run_length_stats' output. Nothing is replaced."""
    hist = rl["histogram_normalised"]
    labels = rl["histogram_bin_labels"]
    n_runs = rl["n_runs"]
    modal_idx = int(np.argmax(hist))
    out = dict(rl)
    out.update({
        "histogram_counts_derived": [int(round(f * n_runs)) for f in hist],
        "fraction_runs_length_1_or_2": float(hist[0] + hist[1]),
        "modal_run_length": labels[modal_idx],
        "modal_run_length_fraction": float(hist[modal_idx]),
        "tail_fraction_at_or_above_20": float(hist[-1]),
    })
    return out


def region_stats(label, t, m, n_blocks, description, with_hd):
    print(f"[{label}] {t.shape[1]}x{t.shape[2]} over {t.shape[0]} codes ...", flush=True)
    rl = enrich(run_length_stats(t, m, n_blocks))
    res = {
        "label": label,
        "description": description,
        "shape": [int(t.shape[1]), int(t.shape[2])],
        "n_codes": int(t.shape[0]),
        "run_length_n_blocks": n_blocks,
        "mask_valid_fraction_mean": float(m.reshape(m.shape[0], -1).mean()),
        "run_lengths": rl,
    }
    if with_hd:
        u, mm = to_bipolar(t, m)
        iu = np.triu_indices(t.shape[0], k=1)
        imp = hd_matrix(u, mm, u, mm)[iu].astype(np.float64)
        d = describe(imp)
        res["impostor_unaligned"] = d
        res["degrees_of_freedom_impostor_unaligned"] = degrees_of_freedom(d["mean"], d["std"])
        print(f"[{label}]   DoF={res['degrees_of_freedom_impostor_unaligned']:.1f}", end="  ")
    print(f"[{label}]   mean run={rl['mean_run_length']:.3f} "
          f"mode={rl['modal_run_length']} frac1-2={rl['fraction_runs_length_1_or_2']:.4f} "
          f"tail>=20={rl['tail_fraction_at_or_above_20']:.4f}")
    return res


def main():
    if _OUT_JSON.exists():
        sys.exit(f"Refusing to overwrite existing {_OUT_JSON} -- move it aside first.")

    ref = json.loads(_REFERENCE_JSON.read_text())
    stored_synth_rl = ref["results"]["synthetic"]["run_lengths"]["mean_run_length"]

    print(f"Loading real codes from {_CASIA_2D_DIR} ...", flush=True)
    gal_t, gal_m, *_ = load_real(None)
    if gal_t.shape[1:] != (32, 512):
        sys.exit(f"Expected (32, 512) codes, got {gal_t.shape[1:]}.")
    print(f"Loading SIC-Gen gallery from {_SYNTH_GALLERY} ...", flush=True)
    s_t, s_m, *_ = load_synthetic(None)
    print()

    dists = {}
    dists["casia_scale0_real"] = region_stats(
        "casia_scale0_real",
        np.ascontiguousarray(gal_t[:, 0:16, 0:256]),
        np.ascontiguousarray(gal_m[:, 0:16, 0:256]), 1,
        "CASIA rows 0-15 (Gabor scale 0, coarse, lambda_phi=28), cols 0-255 = REAL "
        "phase channel. The region closest to SIC-Gen in audit_singlescale_results.json.",
        with_hd=True)
    dists["casia_scale0_imag"] = region_stats(
        "casia_scale0_imag",
        np.ascontiguousarray(gal_t[:, 0:16, 256:512]),
        np.ascontiguousarray(gal_m[:, 0:16, 256:512]), 1,
        "CASIA rows 0-15 (Gabor scale 0, coarse), cols 256-511 = IMAGINARY phase channel. "
        "Included to show whether the choice of phase channel matters.",
        with_hd=True)
    dists["sicgen_full"] = region_stats(
        "sicgen_full", s_t, s_m, 1,
        "SIC-Gen whole code (32x512, 512 genuine angular positions, no phase-channel seam).",
        with_hd=False)

    # Integrity: SIC-Gen run lengths must match the stored bit-level comparison.
    fresh = dists["sicgen_full"]["run_lengths"]["mean_run_length"]
    if abs(fresh - stored_synth_rl) > 1e-9:
        sys.exit(f"SIC-Gen mean run length {fresh} != stored {stored_synth_rl}; aborting.")
    print(f"\n[check] SIC-Gen mean run length {fresh:.4f} matches stored "
          f"{stored_synth_rl:.4f}")

    # Which CASIA phase channel is closer to SIC-Gen?
    def gap(a, b):
        return abs(a - b)
    syn_rl = dists["sicgen_full"]["run_lengths"]["mean_run_length"]
    syn_dof = ref["results"]["synthetic"]["degrees_of_freedom_impostor_unaligned"]
    chan = {}
    for key in ("casia_scale0_real", "casia_scale0_imag"):
        d = dists[key]
        chan[key] = {
            "mean_run_length": d["run_lengths"]["mean_run_length"],
            "gap_to_sicgen_mean_run_length": gap(d["run_lengths"]["mean_run_length"], syn_rl),
            "degrees_of_freedom": d["degrees_of_freedom_impostor_unaligned"],
            "gap_to_sicgen_degrees_of_freedom":
                gap(d["degrees_of_freedom_impostor_unaligned"], syn_dof),
        }
    chan["closer_on_mean_run_length"] = min(
        ("casia_scale0_real", "casia_scale0_imag"),
        key=lambda k: chan[k]["gap_to_sicgen_mean_run_length"])
    chan["closer_on_degrees_of_freedom"] = min(
        ("casia_scale0_real", "casia_scale0_imag"),
        key=lambda k: chan[k]["gap_to_sicgen_degrees_of_freedom"])
    chan["sicgen_mean_run_length"] = syn_rl
    chan["sicgen_degrees_of_freedom"] = syn_dof

    casia_meta = json.loads(_CASIA_MANIFEST.read_text())
    out = {
        "metadata": {
            "purpose": "Plot-ready run-length distributions for the coarse single-scale "
                       "single-phase CASIA region (the real slice closest to SIC-Gen) "
                       "alongside SIC-Gen's, plus the other phase channel of the same "
                       "scale so the channel choice can be checked.",
            "analysis_only": "No re-extraction, no retraining. Reads codes already exported "
                             "by casia-extraction/extract_casia.py and the 2000-subject "
                             "SIC-Gen gallery. No existing result file is modified.",
            "date": date.today().isoformat(),
            "git_commit": get_git_commit(_LEVEL1),
            "related_files": {
                "per_scale_hd_statistics": "results/audit_singlescale_results.json",
                "whole_code_comparison": "results/hd_distribution_real_vs_synthetic.json",
            },
            "phase_channel_convention": {
                "cols_0_255": "REAL part of the Gabor response",
                "cols_256_511": "IMAGINARY part, over the SAME 256 angular positions",
                "source": "iris/nodes/encoder/iris_encoder.py builds each scale as "
                          "np.stack([response.real > 0, response.imag > 0], axis=-1), so "
                          "channel index 0 is the real part; "
                          "christina-fhe-fis/filter_fhe_iris_complete.py extract_iris_code "
                          "concatenates channel 0 then channel 1 along the angular axis, "
                          "putting the real part in cols 0-255.",
                "answer": "The 16x256 regions in audit_singlescale_results.json are the "
                          "REAL phase channel.",
            },
            "scale_convention": {
                "rows_0_15": "Gabor scale 0 -- open-iris pipeline.yaml filter 0, "
                             "lambda_phi=28.0, kernel_size [41,21] (coarse)",
                "rows_16_31": "Gabor scale 1 -- filter 1, lambda_phi=8, kernel_size [17,21] "
                              "(fine)",
            },
            "estimator": "run_length_stats imported unmodified from "
                         "experiments/hd_distribution_real_vs_synthetic.py: runs within "
                         "maximal contiguous valid-bit segments, mask-aware (an occluded bit "
                         "terminates a run and is not counted), non-circular, walked one "
                         "phase channel at a time. Degrees of freedom via mu(1-mu)/sigma^2 on "
                         "the unaligned impostor distribution, all pairs, no sampling.",
            "histogram_format": "histogram_normalised[i] is the fraction of runs of length "
                                "histogram_bin_labels[i]; bins are 1..19 then '>=20' pooled. "
                                "Fractions sum to 1 and are directly comparable across "
                                "distributions despite very different n_runs. "
                                "histogram_counts_derived is round(fraction * n_runs), given "
                                "for convenience only -- the fractions are authoritative.",
            "real": {
                "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
                "open_iris_version": casia_meta.get("open_iris_version"),
                "identity_count_usable": int(gal_t.shape[0]),
                "codes_used": "gallery only (stem '1', image index 00)",
                "excluded_identities": ["524_L", "668_L"],
                "seeds_used": None,
            },
            "synthetic": {
                "gallery_path": str(_SYNTH_GALLERY.relative_to(_ROOT)),
                "subject_count": int(s_t.shape[0]),
                "seed": ref["metadata"]["synthetic"].get("seed"),
                "codes_used": "IC1 (1_template.txt) of each subject",
            },
            "caveats": [
                "The '>=20' bin pools all longer runs: 0.8% of runs for the CASIA region and "
                "0.4% for SIC-Gen. mean_run_length is computed from uncapped totals and is "
                "unaffected by the cap, so a histogram drawn from these bins will slightly "
                "understate the tail relative to the reported mean.",
                "The CASIA regions are 4096 bits per code against SIC-Gen's 16384, and carry "
                "~0.65 mask valid fraction against SIC-Gen's 0.94, so the CASIA histograms "
                "rest on fewer valid bits per code even though the total run count is large.",
                "Structural match is approximate, not exact: SIC-Gen is a single continuous "
                "512-angle axis, while these CASIA regions are 256 angles from one phase "
                "channel of one Gabor scale. Numerical proximity of the run-length "
                "distributions is not evidence the two codes are built the same way.",
            ],
        },
        "distributions": dists,
        "phase_channel_comparison": chan,
    }

    _OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {_OUT_JSON}")


if __name__ == "__main__":
    main()
