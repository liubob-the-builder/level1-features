#!/usr/bin/env python3
"""Does the Gabor scale count / phase-channel layout explain the real-vs-synthetic bit gap?

hd_distribution_real_vs_synthetic.py measured real CASIA codes at 622 DoF with a
mean run length of 4.07, against SIC-Gen's 229 DoF and 6.01. Two structural
differences between the two code formats are candidate explanations:

  - real codes stack TWO Gabor scales (rows 0-15 = filter 0, lambda_phi=28,
    coarse; rows 16-31 = filter 1, lambda_phi=8, fine), while a SIC-Gen code is
    one homogeneous texture;
  - real codes carry TWO phase channels side by side (cols 0-255 = real part,
    cols 256-511 = imaginary part, over the SAME 256 angular positions), while
    SIC-Gen's 512 columns are 512 genuine angles with no channel seam.

Both would inflate the apparent number of independent bits: independent scales
add DoF, and the quadrature phase channel is near-independent of the real part
(measured agreement 0.5009 between column j and column j+256), so pairing them
roughly doubles the effective sample without doubling the underlying signal.

This script tests that by re-measuring the SAME statistics with the SAME
estimators on progressively smaller slices of the real code -- analysis only, on
the codes already exported to casia-extraction/casia-codes-2d/. Nothing is
re-extracted and no existing file is touched.

Regions (all on the real gallery set used by the bit-level comparison):

  full           rows  0-31, cols 0-511  -- must reproduce the stored 622 DoF / 4.07
  scale0_16x512  rows  0-15, cols 0-511  -- one Gabor scale, both phase channels
  scale0_16x256  rows  0-15, cols 0-255  -- one Gabor scale, one phase channel
  scale1_16x512  rows 16-31, cols 0-511  -- the other scale, both phase channels
  scale1_16x256  rows 16-31, cols 0-255  -- the other scale, one phase channel

Every estimator is imported UNMODIFIED from hd_distribution_real_vs_synthetic:
degrees_of_freedom (mu(1-mu)/sigma^2 on the unaligned impostor distribution),
run_length_stats (runs within maximal contiguous valid-bit segments, mask-aware,
non-circular, walked one phase channel at a time), and the bipolar-GEMM
all-pairs HD path. n_blocks is 2 for a 512-column region (two phase channels, so
no run crosses the seam) and 1 for a 256-column region (a single channel).

Impostor distances are unaligned and exhaustive over all gallery pairs,
restricted to the region -- unaligned is the distribution the DoF estimator is
meant for, so no rotation is applied and the rotation convention does not enter.

Usage:
    python level1-features/experiments/audit_singlescale_regions.py
    python level1-features/experiments/audit_singlescale_regions.py --quick
"""

import argparse
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
_OUT_JSON = _RESULTS / "audit_singlescale_results.json"
_OUT_MD = _RESULTS / "audit_singlescale_RESULTS.md"

sys.path.insert(0, str(_EXPERIMENTS))

# Estimators and loaders reused verbatim -- this script defines no statistics of
# its own beyond slicing and the two derived run-length summaries.
from hd_distribution_real_vs_synthetic import (  # noqa: E402
    RUNLENGTH_HIST_CAP,
    degrees_of_freedom,
    describe,
    get_git_commit,
    hd_matrix,
    load_real,
    run_length_stats,
    to_bipolar,
    _CASIA_2D_DIR,
    _CASIA_MANIFEST,
    _SYNTH_GALLERY,
)

# (label, row slice, col slice, n_blocks, description)
REGIONS = [
    ("full",          slice(0, 32),  slice(0, 512), 2,
     "whole code: 2 Gabor scales x 2 phase channels"),
    ("scale0_16x512", slice(0, 16),  slice(0, 512), 2,
     "Gabor scale 0 (coarse, lambda_phi=28), both phase channels"),
    ("scale0_16x256", slice(0, 16),  slice(0, 256), 1,
     "Gabor scale 0 (coarse), real phase channel only"),
    ("scale1_16x512", slice(16, 32), slice(0, 512), 2,
     "Gabor scale 1 (fine, lambda_phi=8), both phase channels"),
    ("scale1_16x256", slice(16, 32), slice(0, 256), 1,
     "Gabor scale 1 (fine), real phase channel only"),
]


def impostor_unaligned(templates: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """All-pairs unaligned fractional HD, upper triangle, via the reused GEMM path."""
    u, m = to_bipolar(templates, masks)
    iu = np.triu_indices(templates.shape[0], k=1)
    return hd_matrix(u, m, u, m)[iu].astype(np.float64)


def summarise_runs(rl: dict) -> dict:
    """Derive the two requested run-length summaries from the reused histogram."""
    hist = rl["histogram_normalised"]
    labels = rl["histogram_bin_labels"]
    modal_idx = int(np.argmax(hist))
    return {
        "mean_run_length": rl["mean_run_length"],
        "n_runs": rl["n_runs"],
        "fraction_runs_length_1_or_2": float(hist[0] + hist[1]),
        "modal_run_length": labels[modal_idx],
        "modal_run_length_fraction": float(hist[modal_idx]),
        "histogram_bin_labels": labels,
        "histogram_normalised": hist,
    }


def analyse_region(label, rows, cols, n_blocks, description, gal_t, gal_m):
    t = np.ascontiguousarray(gal_t[:, rows, cols])
    m = np.ascontiguousarray(gal_m[:, rows, cols])
    n_bits = int(t.shape[1] * t.shape[2])

    print(f"[{label}] {t.shape[1]}x{t.shape[2]} = {n_bits} bits, "
          f"n_blocks={n_blocks} ...", flush=True)

    imp = impostor_unaligned(t, m)
    d = describe(imp)
    rl = summarise_runs(run_length_stats(t, m, n_blocks))
    valid_frac = float(m.reshape(m.shape[0], -1).mean())
    dof = degrees_of_freedom(d["mean"], d["std"])

    print(f"[{label}]   impostor mu={d['mean']:.4f} sd={d['std']:.4f} DoF={dof:.1f}  "
          f"run={rl['mean_run_length']:.3f} frac1-2={rl['fraction_runs_length_1_or_2']:.4f} "
          f"mode={rl['modal_run_length']} valid={valid_frac:.4f}")

    return {
        "label": label,
        "description": description,
        "rows": [rows.start, rows.stop],
        "cols": [cols.start, cols.stop],
        "shape": [int(t.shape[1]), int(t.shape[2])],
        "n_bits_per_code": n_bits,
        "n_valid_bits_per_code_mean": float(valid_frac * n_bits),
        "run_length_n_blocks": n_blocks,
        "n_gallery_codes": int(t.shape[0]),
        "n_impostor_pairs": int(imp.size),
        "impostor_unaligned": d,
        "degrees_of_freedom_impostor_unaligned": dof,
        "mask_valid_fraction_mean": valid_frac,
        "run_lengths": rl,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test on the first 100 identities; writes *_quick.* so the "
                         "full results are never overwritten")
    args = ap.parse_args()

    out_json = (_OUT_JSON.with_name(_OUT_JSON.stem + "_quick.json")
                if args.quick else _OUT_JSON)
    out_md = (_OUT_MD.with_name(_OUT_MD.stem + "_quick.md")
              if args.quick else _OUT_MD)
    for p in (out_json, out_md):
        if p.exists():
            sys.exit(f"Refusing to overwrite existing {p} -- move it aside first.")

    print(f"Loading real codes from {_CASIA_2D_DIR} ...", flush=True)
    gal_t, gal_m, *_ = load_real(100 if args.quick else None)
    print(f"  {gal_t.shape[0]} gallery codes of shape {gal_t.shape[1:]}\n")

    if gal_t.shape[1:] != (32, 512):
        sys.exit(f"Expected (32, 512) codes, got {gal_t.shape[1:]}.")

    ref = json.loads(_REFERENCE_JSON.read_text())
    ref_real = ref["results"]["real"]
    ref_synth = ref["results"]["synthetic"]
    synth_rl = summarise_runs(ref_synth["run_lengths"])
    synth = {
        "source": str(_REFERENCE_JSON.relative_to(_ROOT)),
        "gallery_path": ref["metadata"]["synthetic"]["gallery_path"],
        "n_gallery_codes": ref_synth["n_gallery_codes"],
        "shape": [32, 512],
        "impostor_unaligned": ref_synth["impostor_unaligned"],
        "degrees_of_freedom_impostor_unaligned":
            ref_synth["degrees_of_freedom_impostor_unaligned"],
        "mask_valid_fraction_mean": ref_synth["mask_valid_fraction_mean"],
        "run_lengths": synth_rl,
    }

    results = {}
    for label, rows, cols, n_blocks, desc in REGIONS:
        results[label] = analyse_region(label, rows, cols, n_blocks, desc, gal_t, gal_m)

    # --- reproduction check on the full region ------------------------------
    full = results["full"]
    tol_dof, tol_rl = 0.5, 0.005
    repro = {
        "stored_degrees_of_freedom": ref_real["degrees_of_freedom_impostor_unaligned"],
        "recomputed_degrees_of_freedom": full["degrees_of_freedom_impostor_unaligned"],
        "stored_mean_run_length": ref_real["run_lengths"]["mean_run_length"],
        "recomputed_mean_run_length": full["run_lengths"]["mean_run_length"],
        "stored_impostor_unaligned_mean": ref_real["impostor_unaligned"]["mean"],
        "recomputed_impostor_unaligned_mean": full["impostor_unaligned"]["mean"],
        "tolerance_degrees_of_freedom": tol_dof,
        "tolerance_mean_run_length": tol_rl,
    }
    repro["reproduces_stored_real_statistics"] = bool(
        abs(repro["stored_degrees_of_freedom"]
            - repro["recomputed_degrees_of_freedom"]) < tol_dof
        and abs(repro["stored_mean_run_length"]
                - repro["recomputed_mean_run_length"]) < tol_rl)
    print(f"\n[repro] full region vs stored real: DoF {repro['recomputed_degrees_of_freedom']:.2f} "
          f"vs {repro['stored_degrees_of_freedom']:.2f}, run length "
          f"{repro['recomputed_mean_run_length']:.4f} vs {repro['stored_mean_run_length']:.4f} "
          f"-> {'MATCH' if repro['reproduces_stored_real_statistics'] else 'MISMATCH'}")
    if not repro["reproduces_stored_real_statistics"] and not args.quick:
        sys.exit("Full region does not reproduce the stored real statistics -- "
                 "the slices below cannot be trusted; investigate before using them.")

    # --- directional readings ------------------------------------------------
    s_dof = synth["degrees_of_freedom_impostor_unaligned"]
    s_rl = synth["run_lengths"]["mean_run_length"]

    def toward(a, b):
        """Does moving from region a to region b close the gap to synthetic?"""
        out = {}
        for key, getter, target in (
            ("degrees_of_freedom",
             lambda r: r["degrees_of_freedom_impostor_unaligned"], s_dof),
            ("mean_run_length",
             lambda r: r["run_lengths"]["mean_run_length"], s_rl),
        ):
            va, vb = getter(results[a]), getter(results[b])
            ga, gb = abs(va - target), abs(vb - target)
            out[key] = {
                "from": va, "to": vb, "synthetic": target,
                "gap_before": ga, "gap_after": gb,
                "moved_toward_synthetic": bool(gb < ga),
                "overshot_synthetic": bool((va - target) * (vb - target) < 0),
            }
        return out

    directional = {
        "scale_count_full_to_scale0_16x512": toward("full", "scale0_16x512"),
        "phase_channel_scale0_16x512_to_16x256": toward("scale0_16x512", "scale0_16x256"),
        "scale_count_full_to_scale1_16x512": toward("full", "scale1_16x512"),
        "phase_channel_scale1_16x512_to_16x256": toward("scale1_16x512", "scale1_16x256"),
    }

    # Does synthetic match a single scale, or sit between the two scales?
    def bracket(key, getter):
        v0, v1 = getter(results["scale0_16x512"]), getter(results["scale1_16x512"])
        v0s, v1s = getter(results["scale0_16x256"]), getter(results["scale1_16x256"])
        target = s_dof if key == "degrees_of_freedom" else s_rl
        lo, hi = min(v0, v1), max(v0, v1)
        lo_s, hi_s = min(v0s, v1s), max(v0s, v1s)
        return {
            "synthetic": target,
            "scale0_16x512": v0, "scale1_16x512": v1,
            "scale0_16x256": v0s, "scale1_16x256": v1s,
            "between_the_two_scales_16x512": bool(lo < target < hi),
            "between_the_two_scales_16x256": bool(lo_s < target < hi_s),
            "nearest_region": min(
                ((lbl, abs(getter(results[lbl]) - target)) for lbl in results),
                key=lambda kv: kv[1])[0],
            "nearest_region_absolute_gap": min(
                abs(getter(results[lbl]) - target) for lbl in results),
        }

    scale_bracket = {
        "degrees_of_freedom": bracket(
            "degrees_of_freedom", lambda r: r["degrees_of_freedom_impostor_unaligned"]),
        "mean_run_length": bracket(
            "mean_run_length", lambda r: r["run_lengths"]["mean_run_length"]),
    }

    casia_meta = json.loads(_CASIA_MANIFEST.read_text())
    out = {
        "metadata": {
            "purpose": "Test whether the Gabor scale count and the phase-channel layout "
                       "explain the bit-level gap between real CASIA and SIC-Gen codes, by "
                       "re-measuring the same statistics with the same estimators on "
                       "progressively smaller slices of the real (32,512) code.",
            "analysis_only": "No re-extraction, no retraining, no existing file modified. "
                             "Reads the codes already exported by casia-extraction/extract_casia.py.",
            "date": date.today().isoformat(),
            "git_commit": get_git_commit(_LEVEL1),
            "quick_mode": bool(args.quick),
            "subject_limit": 100 if args.quick else None,
            "real": {
                "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
                "casia_manifest": str(_CASIA_MANIFEST.relative_to(_ROOT)),
                "open_iris_version": casia_meta.get("open_iris_version"),
                "identity_count_usable": int(gal_t.shape[0]),
                "subject_selection_rule": casia_meta.get("subject_selection_rule"),
                "seeds_used": None,
                "codes_used": "gallery only (stem '1', image index 00) -- the same set the "
                              "bit-level comparison used for impostor pairs and run lengths",
                "excluded_identities": ["524_L", "668_L"],
            },
            "synthetic_reference": {
                "gallery_path": str(_SYNTH_GALLERY.relative_to(_ROOT)),
                "subject_count": ref_synth["n_gallery_codes"],
                "seed": ref["metadata"]["synthetic"].get("seed"),
                "note": "Not recomputed here -- read from the stored bit-level comparison.",
            },
            "code_layout_verified": {
                "rows_0_15": "Gabor scale 0 (open-iris pipeline.yaml filter 0: "
                             "lambda_phi=28.0, kernel_size [41,21], coarse)",
                "rows_16_31": "Gabor scale 1 (filter 1: lambda_phi=8, kernel_size [17,21], fine)",
                "cols_0_255": "real phase channel",
                "cols_256_511": "imaginary phase channel, over the SAME 256 angular positions",
                "source": "iris/nodes/encoder/iris_encoder.py stacks [real>0, imag>0] on "
                          "axis=-1 giving (16,256,2) per filter; "
                          "christina-fhe-fis/filter_fhe_iris_complete.py extract_iris_code "
                          "concatenates the 2 channels along the angular axis to (16,512) "
                          "then vstacks the 2 filters to (32,512).",
                "empirical_confirmation": "Adjacent-column bit agreement is ~0.88 everywhere "
                                          "except across 255|256 where it is 0.41 (the channel "
                                          "seam); agreement between column j and column j+256 "
                                          "is 0.5009 (real/imag in quadrature, near-independent); "
                                          "raw mean run length is ~8.4 in rows 0-15 vs ~3.1 in "
                                          "rows 16-31 (coarse vs fine filter).",
            },
            "estimators": {
                "degrees_of_freedom": "mu(1-mu)/sigma^2 on the UNALIGNED impostor distribution, "
                                      "imported unmodified from "
                                      "experiments/hd_distribution_real_vs_synthetic.py "
                                      "(itself sic-gen/validation.py's estimator).",
                "run_lengths": "run_length_stats imported unmodified: runs within maximal "
                               "contiguous valid-bit segments, mask-aware (an occluded bit "
                               "terminates a run and is not counted), non-circular, walked one "
                               "phase channel at a time (n_blocks=2 for 512-column regions, "
                               "n_blocks=1 for 256-column regions) so no run crosses the seam.",
                "mask_valid_fraction": "mean of the mask over the region (1 = valid).",
                "impostor_pairs": "unaligned, exhaustive over all gallery pairs, restricted to "
                                  "the region; no sampling and no rotation.",
            },
            "caveats": [
                "Each smaller region contains fewer bits (16384 -> 8192 -> 4096 valid-bit "
                "budget before masking), so per-pair HD is estimated from fewer samples and "
                "the impostor sigma widens partly for that reason alone -- some of the DoF "
                "movement across regions is a finite-sample effect, not a change in the "
                "underlying bit correlation.",
                "None of the real regions structurally matches SIC-Gen. A SIC-Gen code is a "
                "single continuous 512-angle axis with no phase-channel seam; the real "
                "16x512 regions are 256 angles carried in two quadrature channels, and the "
                "real 16x256 regions are 256 angles in one channel. So even an exact "
                "numerical match would not mean the two codes have the same structure.",
                "The synthetic figures are read from the stored bit-level comparison and are "
                "not recomputed here; only the real regions were measured for this file.",
            ],
        },
        "regions_real": results,
        "synthetic_reference_statistics": synth,
        "full_region_reproduction_check": repro,
        "directional_readings": directional,
        "does_synthetic_match_a_single_scale": scale_bracket,
    }

    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_json}")
    write_markdown(out, out_md)
    print(f"Summary written to {out_md}")


def _row(r):
    return (f"| {r['label']} ({r['shape'][0]}x{r['shape'][1]}) "
            f"| {r['n_bits_per_code']} "
            f"| {r['impostor_unaligned']['mean']:.4f} "
            f"| {r['impostor_unaligned']['std']:.4f} "
            f"| {r['degrees_of_freedom_impostor_unaligned']:.1f} "
            f"| {r['run_lengths']['mean_run_length']:.3f} "
            f"| {r['run_lengths']['fraction_runs_length_1_or_2']:.3f} "
            f"| {r['run_lengths']['modal_run_length']} "
            f"| {r['mask_valid_fraction_mean']:.3f} |")


def write_markdown(out, path):
    res = out["regions_real"]
    syn = out["synthetic_reference_statistics"]
    repro = out["full_region_reproduction_check"]
    d = out["directional_readings"]
    br = out["does_synthetic_match_a_single_scale"]
    m = out["metadata"]

    def arrow(block):
        return "toward synthetic" if block["moved_toward_synthetic"] else "away from synthetic"

    L = []
    L.append("# Do Gabor scale count and phase-channel layout explain the real-vs-synthetic bit gap?")
    L.append("")
    L.append(f"`experiments/audit_singlescale_regions.py` | {m['date']} | "
             f"commit `{m['git_commit'][:12]}` | "
             f"{res['full']['n_gallery_codes']} real gallery codes"
             + (" | **quick mode**" if m["quick_mode"] else ""))
    L.append("")
    L.append("Analysis only: no re-extraction, no retraining, no existing file modified. "
             "Every estimator is imported unmodified from "
             "`experiments/hd_distribution_real_vs_synthetic.py`.")
    L.append("")
    L.append("## Verified code layout")
    L.append("")
    L.append("In a real (32, 512) code: **rows 0-15 = Gabor scale 0** (coarse, `lambda_phi=28`), "
             "**rows 16-31 = Gabor scale 1** (fine, `lambda_phi=8`); "
             "**cols 0-255 = real phase channel**, **cols 256-511 = imaginary**, both over the "
             "same 256 angular positions. Confirmed from `iris/nodes/encoder/iris_encoder.py` "
             "and `extract_iris_code`, and empirically: adjacent-column agreement is ~0.88 "
             "everywhere except across 255|256 where it is 0.41, and agreement between column "
             "*j* and column *j*+256 is 0.5009 (quadrature, near-independent).")
    L.append("")
    L.append("## Reproduction check")
    L.append("")
    verdict = "reproduced" if repro["reproduces_stored_real_statistics"] else "NOT reproduced"
    L.append(f"The full 32x512 region gives **{repro['recomputed_degrees_of_freedom']:.1f} DoF** "
             f"and **mean run length {repro['recomputed_mean_run_length']:.3f}** against the "
             f"stored {repro['stored_degrees_of_freedom']:.1f} / "
             f"{repro['stored_mean_run_length']:.3f} -- **{verdict}**. "
             + ("The slices below can be read against this baseline."
                if repro["reproduces_stored_real_statistics"]
                else "The slices below should NOT be trusted."))
    L.append("")
    L.append("## All regions vs stored SIC-Gen")
    L.append("")
    L.append("| Region | Bits | Impostor mean HD | Impostor sigma | DoF | Mean run length "
             "| Frac. runs 1-2 | Modal run | Mask valid |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lbl in ("full", "scale0_16x512", "scale0_16x256", "scale1_16x512", "scale1_16x256"):
        L.append(_row(res[lbl]))
    L.append(f"| **SIC-Gen (stored, 32x512)** | 16384 "
             f"| {syn['impostor_unaligned']['mean']:.4f} "
             f"| {syn['impostor_unaligned']['std']:.4f} "
             f"| {syn['degrees_of_freedom_impostor_unaligned']:.1f} "
             f"| {syn['run_lengths']['mean_run_length']:.3f} "
             f"| {syn['run_lengths']['fraction_runs_length_1_or_2']:.3f} "
             f"| {syn['run_lengths']['modal_run_length']} "
             f"| {syn['mask_valid_fraction_mean']:.3f} |")
    L.append("")
    L.append("Region key: `scale0` = rows 0-15 (coarse), `scale1` = rows 16-31 (fine); "
             "`16x512` = both phase channels, `16x256` = real channel only.")
    L.append("")
    L.append("## (a) Does dropping to one scale (32x512 -> 16x512) move toward synthetic?")
    L.append("")
    for lbl, key in (("scale 0 (coarse)", "scale_count_full_to_scale0_16x512"),
                     ("scale 1 (fine)", "scale_count_full_to_scale1_16x512")):
        b = d[key]
        L.append(f"- **{lbl}:** DoF {b['degrees_of_freedom']['from']:.1f} -> "
                 f"{b['degrees_of_freedom']['to']:.1f} ({arrow(b['degrees_of_freedom'])}"
                 + (", overshoots" if b['degrees_of_freedom']['overshot_synthetic'] else "")
                 + f"); mean run length {b['mean_run_length']['from']:.3f} -> "
                 f"{b['mean_run_length']['to']:.3f} ({arrow(b['mean_run_length'])}"
                 + (", overshoots" if b['mean_run_length']['overshot_synthetic'] else "") + ").")
    L.append("")
    L.append("## (b) Does dropping the second phase channel (16x512 -> 16x256) move it further?")
    L.append("")
    for lbl, key in (("scale 0 (coarse)", "phase_channel_scale0_16x512_to_16x256"),
                     ("scale 1 (fine)", "phase_channel_scale1_16x512_to_16x256")):
        b = d[key]
        L.append(f"- **{lbl}:** DoF {b['degrees_of_freedom']['from']:.1f} -> "
                 f"{b['degrees_of_freedom']['to']:.1f} ({arrow(b['degrees_of_freedom'])}"
                 + (", overshoots" if b['degrees_of_freedom']['overshot_synthetic'] else "")
                 + f"); mean run length {b['mean_run_length']['from']:.3f} -> "
                 f"{b['mean_run_length']['to']:.3f} ({arrow(b['mean_run_length'])}"
                 + (", overshoots" if b['mean_run_length']['overshot_synthetic'] else "") + ").")
    L.append("")
    L.append("## Does SIC-Gen match a single real scale, or sit between the two?")
    L.append("")
    for key, name, fmt in (("degrees_of_freedom", "Degrees of freedom", "{:.1f}"),
                           ("mean_run_length", "Mean run length", "{:.3f}")):
        b = br[key]
        between512 = ("**between the two scales**" if b["between_the_two_scales_16x512"]
                      else "**outside the range spanned by the two scales**")
        between256 = ("between them" if b["between_the_two_scales_16x256"] else "outside them")
        L.append(f"- **{name}:** SIC-Gen at {fmt.format(b['synthetic'])} against "
                 f"scale 0 = {fmt.format(b['scale0_16x512'])} and "
                 f"scale 1 = {fmt.format(b['scale1_16x512'])} at 16x512 -- {between512}. "
                 f"At 16x256 (scale 0 = {fmt.format(b['scale0_16x256'])}, "
                 f"scale 1 = {fmt.format(b['scale1_16x256'])}) it falls {between256}. "
                 f"Closest single region: `{b['nearest_region']}` "
                 f"(gap {fmt.format(b['nearest_region_absolute_gap'])}).")
    L.append("")
    L.append("## Caveats")
    L.append("")
    for c in m["caveats"]:
        L.append(f"- {c}")
    L.append("")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
