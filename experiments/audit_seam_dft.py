#!/usr/bin/env python3
"""Audit: how much does the 255|256 channel seam affect the real-data DFT feature?

The DFT Level-1 feature takes ONE 512-point rFFT per row of a 32x512 code. That is
correct for SIC-Gen codes, whose 512 columns are 512 true angular positions. It is
NOT correct for real Open Iris codes: there, extract_iris_code() concatenates the
two Gabor phase channels along the column axis, so columns 0-255 and 256-511 are the
SAME 256 angular positions twice over. The true angular period is 256, and a
512-point transform runs straight across the channel seam at 255|256.

That seam is a measured discontinuity, not a theoretical one: mean adjacent-column
bit agreement over 400 real identities is 0.810 in the interior of a channel but
0.328 across 255|256, while SIC-Gen shows no such break (0.830 vs 0.836).

This script quantifies the consequence for the real-data DFT numbers by building a
per-channel variant -- two separate 256-point rFFTs per row, one per channel, never
transforming across the seam -- and evaluating it on the identical real protocol as
the stored result.

READ-ONLY on every existing script, feature module and result file. All functions
that define the stored protocol are imported unmodified; nothing existing is edited
or overwritten. New outputs are prefixed audit_seam_.

VARIANTS EVALUATED
------------------
  baseline_512          Recomputation of the stored feature (512-point rFFT, bins
                        30-50, 21-dim). Included as a self-check that this script
                        reproduces myfeatures_realcasia_full2000.json.

  (a) freq_matched      Per-channel 256-point rFFT, bins 15-25.
                        Bin k of an N-point rFFT is k cycles per N samples. One
                        256-column channel spans one full revolution, so per-channel
                        bin j is j cycles/revolution; in the 512-point transform of
                        the two-block concatenation that same component lands at bin
                        2j. Hence j_256 = k_512 / 2, and the stored band 30-50 maps
                        to 15-25 (11 bins, half the frequency resolution).
                        Exact only for the component COMMON to both channels; the
                        odd 512-bins carry the channel-DIFFERENCE component, which
                        has no per-channel counterpart.

  (b) reselected        Per-channel 256-point rFFT, band chosen by rerunning Stage
                        1's per-bin cross-code variance procedure directly on the
                        per-channel spectra. Selected on TRAIN-SPLIT identities only
                        (nn_identity_split.json, subject-disjoint from test) to avoid
                        selection-on-test.

Each per-channel variant is built two ways, to separate "seam removed" from
"dimension doubled":
    concat  -- the two channels' kept bins concatenated  -> 2 * n_bins
    avg     -- the two channels' spectra averaged first  -> n_bins

EVALUATION
----------
Full protocol: identical to eval_myfeatures_realcasia_full2000.py -- 1998 galleries
(stem '1', gallery_ok only), 16457 probes (stems '2'-'10'), SSD, dense-rank
tie-break, same K values.

Test-split protocol: probes restricted to the 299 held-out test identities of
nn_identity_split.json, ranked against the same full 1998-identity gallery. This is
the clean read for variant (b), whose band was selected on the train split; it is
reported for every variant so the comparison stays like-for-like.

EER/d' are deliberately not recomputed: the stored values use full impostor sets, and
they are not needed for the seam question. Recall@K and rank are.

Usage:
    python level1-features/experiments/audit_seam_dft.py
    python level1-features/experiments/audit_seam_dft.py --limit 100   # smoke test
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
_FEATURES = _LEVEL1 / "features"
_CASIA_DIR = _ROOT / "casia-extraction"
_CASIA_2D_DIR = _CASIA_DIR / "casia-codes-2d"
_RESULTS = _LEVEL1 / "results"
_STORED_REAL_PATH = _RESULTS / "myfeatures_realcasia_full2000.json"
_SPLIT_PATH = _RESULTS / "nn_identity_split.json"
_OUT_JSON = _RESULTS / "audit_seam_dft_results.json"
_OUT_MD = _RESULTS / "audit_seam_dft_RESULTS.md"
_OUT_PNG = _RESULTS / "audit_seam_dft_variance.png"

sys.path.insert(0, str(_FEATURES))
sys.path.insert(0, str(_EXPERIMENTS))

# --- imported unmodified; nothing below edits these modules -------------------
from dft_spectrum import (  # noqa: E402
    row_magnitude_spectrum, code_magnitude_spectrum, load_template, MIN_VALID_FRAC,
)
from stage2_evaluate import ssd_matrix  # noqa: E402
from eval_myfeatures_realcasia_full2000 import (  # noqa: E402
    load_manifest, available_probe_stems, load_mask, dense_ranks, recall_at_k,
    get_git_commit, K_VALUES, EXPECTED_N_GALLERY,
)

DFT_BINS_512 = np.arange(30, 51)        # the stored band, on the 512-point transform
N_CHANNELS = 2
CHANNEL_WIDTH = 256
N_FREQS_CHANNEL = CHANNEL_WIDTH // 2 + 1   # 129
N_FREQS_FULL = 512 // 2 + 1                # 257
PROBE_CHUNK = 4000                          # probe rows per ranking chunk


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def per_channel_spectra(template, mask, min_valid_frac=MIN_VALID_FRAC):
    """Mask-aware row-averaged magnitude spectrum for each 256-column channel.

    Mirrors dft_spectrum.code_magnitude_spectrum's mask-aware 'average' path
    (its lines 99-113) exactly, but applied independently to each channel block:
      - a (row, channel) block whose valid fraction is < min_valid_frac is dropped
        from that channel's average;
      - a kept block has its occluded bits mean-filled with that block's own
        valid-bit mean before the rFFT;
      - if every block of a channel is dropped, fall back to all rows for that
        channel (same degenerate fallback as the original).

    Returns:
        (N_CHANNELS, N_FREQS_CHANNEL) float64.
    """
    out = np.empty((N_CHANNELS, N_FREQS_CHANNEL), dtype=np.float64)
    for c in range(N_CHANNELS):
        lo, hi = c * CHANNEL_WIDTH, (c + 1) * CHANNEL_WIDTH
        blk_t, blk_m = template[:, lo:hi], mask[:, lo:hi]
        kept = []
        for r in range(blk_t.shape[0]):
            valid = blk_m[r].astype(bool)
            valid_frac = valid.mean() if valid.size else 0.0
            if valid_frac < min_valid_frac:
                continue
            row = blk_t[r].astype(np.float64).copy()
            if valid_frac < 1.0:
                row[~valid] = row[valid].mean()
            kept.append(row_magnitude_spectrum(row))
        if not kept:
            kept = [row_magnitude_spectrum(blk_t[r].astype(np.float64))
                    for r in range(blk_t.shape[0])]
        out[c] = np.stack(kept).mean(axis=0)
    return out


def extract_all(identities, limit=None):
    """One pass over the data; store FULL spectra so any bin set can be sliced later."""
    gallery_ids, probe_true_ids = [], []
    g_full, g_chan, p_full, p_chan = [], [], [], []

    n_done = 0
    for ident in identities:
        if not ident["gallery_ok"]:
            continue
        if limit and n_done >= limit:
            break
        identity = ident["identity"]
        d = _CASIA_2D_DIR / identity

        t, m = load_template(d, "1"), load_mask(d, "1")
        g_full.append(code_magnitude_spectrum(t, row_agg="average", mask=m))
        g_chan.append(per_channel_spectra(t, m))
        gallery_ids.append(identity)

        for stem in available_probe_stems(d):
            tp, mp = load_template(d, stem), load_mask(d, stem)
            p_full.append(code_magnitude_spectrum(tp, row_agg="average", mask=mp))
            p_chan.append(per_channel_spectra(tp, mp))
            probe_true_ids.append(identity)

        n_done += 1
        if n_done % 200 == 0:
            print(f"  ... {n_done} identities, {len(probe_true_ids)} probes", flush=True)

    return {
        "gallery_ids": gallery_ids,
        "probe_true_ids": probe_true_ids,
        "gallery_full": np.stack(g_full),            # (n_gal, 257)
        "gallery_chan": np.stack(g_chan),            # (n_gal, 2, 129)
        "probe_full": np.stack(p_full),              # (n_probe, 257)
        "probe_chan": np.stack(p_chan),              # (n_probe, 2, 129)
    }


def build_vectors(chan_spectra, bins, mode):
    """chan_spectra: (n, 2, 129) -> feature matrix.

    mode 'concat' -> (n, 2 * len(bins));  mode 'avg' -> (n, len(bins)).
    """
    if mode == "concat":
        return chan_spectra[:, :, bins].reshape(chan_spectra.shape[0], -1)
    if mode == "avg":
        return chan_spectra.mean(axis=1)[:, bins]
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Bin selection (Stage 1's procedure, rerun on per-channel spectra)
# ---------------------------------------------------------------------------

def select_band(spectra, width, name, criterion="variance"):
    """Stage 1's per-bin cross-code statistic, then a contiguous band of fixed width.

    spectra: (n_codes, n_freqs). DC (bin 0) is dropped before the statistic is taken,
    exactly as stage1_spectrum_inspection.py does.

    criterion:
      'variance' -- raw cross-code variance. This is Stage 1's literal criterion and
                    is reported as the faithful rerun. NOTE: raw variance is scale
                    dependent, and magnitude falls off with frequency, so on real
                    codes it is dominated by the lowest bins. On the synthetic data
                    Stage 1 was run on this did not bite, because SIC-Gen has a strong
                    genuine spectral peak at bins 38-41 where the mean and the variance
                    peaks coincide.
      'cov'      -- variance / mean^2 (squared coefficient of variation), which removes
                    that scale dependence and asks where codes differ most RELATIVE to
                    the local magnitude. Not Stage 1's criterion; reported because the
                    raw-variance answer is degenerate on real data.
      'mean'     -- peak of the mean magnitude spectrum, i.e. where the texture energy
                    actually sits. Stage 1 plotted this alongside the variance.

    The width is fixed to match variant (a)'s width so the ONLY difference between (a)
    and (b) is where the band sits, not how wide it is. The band is centred on the peak
    bin and clamped to the valid range; clamping is reported.
    """
    mean_spec = spectra.mean(axis=0)
    var_spec = spectra.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov_spec = np.where(mean_spec > 0, var_spec / np.square(mean_spec), 0.0)

    stat = {"variance": var_spec, "cov": cov_spec, "mean": mean_spec}[criterion]
    stat_no_dc = stat[1:]
    peak_bin = int(np.argmax(stat_no_dc)) + 1

    half = width // 2
    lo = max(1, peak_bin - half)
    hi = min(spectra.shape[1] - 1, lo + width - 1)
    lo = max(1, hi - width + 1)
    clamped = (peak_bin - half < 1) or (peak_bin + half > spectra.shape[1] - 1)
    bins = np.arange(lo, hi + 1)
    top20 = (np.argsort(stat_no_dc)[::-1][:20] + 1).tolist()

    print(f"  [{name}/{criterion}] peak bin = {peak_bin}; band = {lo}-{hi} "
          f"({len(bins)} bins){' [CLAMPED]' if clamped else ''}; "
          f"top-5 = {top20[:5]}")
    return {
        "criterion": criterion,
        "peak_bin": peak_bin,
        "band_lo": int(lo), "band_hi": int(hi),
        "band_clamped_at_spectrum_edge": bool(clamped),
        "bins": bins.tolist(),
        "top20_bins": top20,
        "peak_bin_variance": int(np.argmax(var_spec[1:])) + 1,
        "peak_bin_cov": int(np.argmax(cov_spec[1:])) + 1,
        "peak_bin_mean": int(np.argmax(mean_spec[1:])) + 1,
        "variance_profile": var_spec.tolist(),
        "mean_profile": mean_spec.tolist(),
        "cov_profile": cov_spec.tolist(),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(gallery, probe, true_col, k_values=K_VALUES):
    """Rank each probe against the full gallery by SSD. Mirrors the stored protocol."""
    n_probe = probe.shape[0]
    ranks = np.empty(n_probe, dtype=np.int64)
    for lo in range(0, n_probe, PROBE_CHUNK):
        hi = min(lo + PROBE_CHUNK, n_probe)
        D = ssd_matrix(gallery, probe[lo:hi])           # (chunk, n_gallery)
        rm = dense_ranks(D)                             # row-wise, so chunking is exact
        ranks[lo:hi] = rm[np.arange(hi - lo), true_col[lo:hi]] + 1
        del D, rm
    out = {"dim": int(gallery.shape[1]), "n_gallery": int(gallery.shape[0]),
           "n_probe": int(n_probe),
           "median_rank": float(np.median(ranks)), "mean_rank": float(ranks.mean())}
    for k in k_values:
        out[f"recall@{k}"] = recall_at_k(ranks, k)
    return out


def deltas(variant, stored, k_values=K_VALUES):
    d = {"median_rank": variant["median_rank"] - stored["median_rank"]}
    for k in k_values:
        key = f"recall@{k}"
        if key in stored:
            d[key] = variant[key] - stored[key]
    return d


def classify(delta_r50, delta_r10):
    """State the effect size rather than only its direction."""
    m = max(abs(delta_r50), abs(delta_r10))
    if m < 0.005:
        return "negligible"
    if m < 0.02:
        return "small"
    if m < 0.05:
        return "moderate"
    return "material"


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only this many gallery identities")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    for p in (_OUT_JSON, _OUT_MD):
        if p.exists():
            sys.exit(f"Refusing to overwrite existing {p} -- move it aside first.")

    stored_all = json.loads(_STORED_REAL_PATH.read_text())
    stored = stored_all["results_real_full2000"]["DFT"]
    split = json.loads(_SPLIT_PATH.read_text())
    train_ids = set(split["identities"]["train"])
    test_ids = set(split["identities"]["test"])
    print(f"Stored 512-point DFT: R@10={stored['recall@10']:.4f} "
          f"R@50={stored['recall@50']:.4f} median_rank={stored['median_rank']:.1f} "
          f"dim={stored['dim']}")
    print(f"Split: {len(train_ids)} train identities (bin selection), "
          f"{len(test_ids)} test identities (clean read)")

    manifest = load_manifest()
    print(f"\nExtracting spectra from {_CASIA_2D_DIR} ...", flush=True)
    data = extract_all(manifest["identities"], limit=args.limit)

    gallery_ids = data["gallery_ids"]
    probe_true_ids = data["probe_true_ids"]
    n_gallery, n_probe = len(gallery_ids), len(probe_true_ids)
    print(f"Extracted {n_gallery} galleries, {n_probe} probes.")
    if args.limit is None and n_gallery != EXPECTED_N_GALLERY:
        sys.exit(f"Expected {EXPECTED_N_GALLERY} galleries, got {n_gallery}.")

    gallery_idx = {g: i for i, g in enumerate(gallery_ids)}
    true_col = np.array([gallery_idx[p] for p in probe_true_ids])
    test_probe_mask = np.array([p in test_ids for p in probe_true_ids])
    print(f"Test-split probes: {int(test_probe_mask.sum())} of {n_probe}")

    # ---- (b) bin selection, on TRAIN-SPLIT gallery codes only ----------------
    train_rows = np.array([i for i, g in enumerate(gallery_ids) if g in train_ids])
    print(f"\nBin selection on {len(train_rows)} train-split gallery codes "
          f"(subject-disjoint from test):")
    width = len(DFT_BINS_512) // 2 + 1                  # 21 -> 11, matches variant (a)
    train_chan = data["gallery_chan"][train_rows]
    train_chan_avg = train_chan.mean(axis=1)

    selections = {}
    for crit in ("variance", "cov", "mean"):
        selections[crit] = {
            "channel_0": select_band(train_chan[:, 0, :], width, "ch0", crit),
            "channel_1": select_band(train_chan[:, 1, :], width, "ch1", crit),
            "channel_averaged": select_band(train_chan_avg, width, "ch-avg", crit),
        }
    # Context: where the same statistic peaks on the 512-point transform of REAL
    # codes, versus the stored band 30-50 that was selected on synthetic data.
    sel_full = select_band(data["gallery_full"][train_rows], len(DFT_BINS_512),
                           "512-point (context)", "variance")

    freq_matched_bins = np.arange(DFT_BINS_512[0] // 2, DFT_BINS_512[-1] // 2 + 1)
    band = {c: np.array(selections[c]["channel_averaged"]["bins"]) for c in selections}
    print(f"\n(a) frequency-matched : bins {freq_matched_bins[0]}-{freq_matched_bins[-1]}")
    for c in ("variance", "cov", "mean"):
        print(f"(b) {c:<9} band    : bins {band[c][0]}-{band[c][-1]}")

    # ---- Build every variant ------------------------------------------------
    variants = {
        "baseline_512": (data["gallery_full"][:, DFT_BINS_512],
                         data["probe_full"][:, DFT_BINS_512]),
    }
    for label, bins in (("a_freq_matched", freq_matched_bins),
                        ("b_var", band["variance"]),
                        ("b_cov", band["cov"]),
                        ("b_meanpeak", band["mean"])):
        for mode in ("concat", "avg"):
            variants[f"{label}_{mode}"] = (
                build_vectors(data["gallery_chan"], bins, mode),
                build_vectors(data["probe_chan"], bins, mode))

    # ---- Evaluate -----------------------------------------------------------
    results_full, results_test = {}, {}
    print("\nEvaluating (full protocol) ...", flush=True)
    for name, (g, pr) in variants.items():
        results_full[name] = evaluate(g, pr, true_col)
        r = results_full[name]
        print(f"  {name:<22} dim={r['dim']:>3}  R@10={r['recall@10']:.4f}  "
              f"R@50={r['recall@50']:.4f}  R@100={r['recall@100']:.4f}  "
              f"R@300={r['recall@300']:.4f}  med={r['median_rank']:.1f}", flush=True)

    print("\nEvaluating (test-split probes, full gallery) ...", flush=True)
    for name, (g, pr) in variants.items():
        results_test[name] = evaluate(g, pr[test_probe_mask], true_col[test_probe_mask])
        r = results_test[name]
        print(f"  {name:<22} dim={r['dim']:>3}  R@10={r['recall@10']:.4f}  "
              f"R@50={r['recall@50']:.4f}  med={r['median_rank']:.1f}", flush=True)

    # ---- Self-check: does baseline_512 reproduce the stored numbers? --------
    base = results_full["baseline_512"]
    repro = {k: {"recomputed": base[k], "stored": stored[k],
                 "abs_diff": abs(base[k] - stored[k])}
             for k in ("recall@10", "recall@50", "recall@100", "recall@300",
                       "median_rank", "mean_rank") if k in stored}
    max_recall_diff = max(v["abs_diff"] for k, v in repro.items() if k.startswith("recall"))
    if args.limit is not None:
        repro_status = (f"not applicable -- --limit {args.limit} was used, so this run "
                        f"evaluates a {n_gallery}-identity subset, not the stored "
                        f"1998-identity protocol")
        print(f"\nSelf-check vs stored: skipped ({repro_status})")
    else:
        repro_status = ("reproduced" if max_recall_diff < 1e-9
                        else "MISMATCH -- investigate before trusting the deltas")
        print(f"\nSelf-check vs stored: max |recall diff| = {max_recall_diff:.2e} "
              f"({repro_status})")
    repro["status"] = repro_status
    repro["max_abs_recall_diff"] = max_recall_diff

    # ---- Verdicts -----------------------------------------------------------
    verdicts = {}
    for name in variants:
        if name == "baseline_512":
            continue
        d = deltas(results_full[name], stored)
        verdicts[name] = {
            "delta_vs_stored_full_protocol": d,
            "effect_size": classify(d["recall@50"], d["recall@10"]),
        }

    out = {
        "metadata": {
            "purpose": "Quantify how much the 255|256 Gabor phase-channel seam affects "
                       "the real-data DFT Level-1 feature, by comparing the stored "
                       "512-point transform against per-channel 256-point variants that "
                       "never transform across the seam.",
            "date": date.today().isoformat(),
            "git_commit": get_git_commit(_LEVEL1),
            "read_only": "No existing script, feature module or result file is modified. "
                         "code_magnitude_spectrum/row_magnitude_spectrum/load_template/"
                         "MIN_VALID_FRAC, ssd_matrix, and load_manifest/"
                         "available_probe_stems/load_mask/dense_ranks/recall_at_k/K_VALUES "
                         "are imported unmodified.",
            "data_source": str(_CASIA_2D_DIR.relative_to(_ROOT)),
            "casia_manifest": str((_CASIA_DIR / "manifest.json").relative_to(_ROOT)),
            "stored_result_compared_against": str(_STORED_REAL_PATH.relative_to(_ROOT))
                                              + " -> results_real_full2000.DFT",
            "n_gallery": n_gallery,
            "n_probe": n_probe,
            "n_probe_test_split": int(test_probe_mask.sum()),
            "n_train_codes_for_bin_selection": int(len(train_rows)),
            "subject_limit": args.limit,
            "seam_evidence": "Mean adjacent-column bit agreement over 400 real "
                             "identities: 0.810 interior, 0.328 across 255|256, 0.692 at "
                             "511|0. SIC-Gen shows no break (0.836 / 0.830 / 0.706).",
            "row_drop_rule": f"Per-256-block: a (row, channel) block with valid fraction "
                             f"< {MIN_VALID_FRAC} is dropped from that channel's average; "
                             "kept blocks are mean-filled with the block's own valid-bit "
                             "mean before the rFFT. Faithful analogue of the stored "
                             "feature's per-row rule.",
            "bin_mapping_a": "Bin k of an N-point rFFT is k cycles per N samples. A "
                             "256-column channel spans one revolution, so per-channel bin "
                             "j is j cycles/revolution and appears at bin 2j of the "
                             "512-point transform of the concatenation. Hence "
                             "j_256 = k_512/2, mapping the stored band 30-50 to 15-25. "
                             "Exact for the component common to both channels; the odd "
                             "512-bins carry the channel-difference component, which has "
                             "no per-channel counterpart.",
            "bin_selection_b": f"Stage 1's procedure rerun on the per-channel spectra "
                               f"using ONLY the {len(train_rows)} train-split gallery "
                               f"codes of nn_identity_split.json (subject-disjoint from "
                               f"test), to avoid selection-on-test. Band width fixed at "
                               f"{width} bins to match variant (a), so only the band's "
                               f"POSITION differs, centred on the peak bin. Three "
                               f"criteria are reported: 'variance' is Stage 1's literal "
                               f"criterion; 'cov' (variance/mean^2) removes the "
                               f"magnitude-falloff bias that makes raw variance "
                               f"degenerate on real codes; 'mean' is the mean-spectrum "
                               f"peak, which Stage 1 plotted alongside the variance and "
                               f"which coincided with it on synthetic data.",
            "metrics_note": "EER and d' are deliberately not recomputed: the stored values "
                            "use full impostor sets and are not needed for the seam "
                            "question. Recall@K and rank are recomputed under the stored "
                            "protocol exactly.",
            "protocols": {
                "full": "1998 galleries (stem '1', gallery_ok only), 16457 probes "
                        "(stems '2'-'10'), SSD, dense-rank tie-break -- identical to "
                        "eval_myfeatures_realcasia_full2000.py.",
                "test_split": "Same gallery; probes restricted to the 299 held-out test "
                              "identities. The clean read for variant (b).",
            },
        },
        "bin_selection": {
            "width_used": int(width),
            "frequency_matched_bins": freq_matched_bins.tolist(),
            "bands_used_by_criterion": {c: band[c].tolist() for c in band},
            "selections": selections,
            "full_512_context": sel_full,
        },
        "stored_512_reference": stored,
        "baseline_reproduction_check": repro,
        "results_full_protocol": results_full,
        "results_test_split": results_test,
        "verdicts_vs_stored": verdicts,
    }
    _OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {_OUT_JSON}")

    write_markdown(out, width, freq_matched_bins, band, selections, sel_full, stored,
                   results_full, results_test, verdicts, repro)
    print(f"Summary written to {_OUT_MD}")

    if not args.no_plot:
        make_plot(selections, sel_full, freq_matched_bins, band)
        print(f"Plot saved to {_OUT_PNG}")


LABELS = {
    "baseline_512": "Recomputed 512-point (self-check)",
    "a_freq_matched_concat": "(a) freq-matched 15-25, channel-concat",
    "a_freq_matched_avg": "(a) freq-matched 15-25, channel-averaged",
    "b_var_concat": "(b) re-selected [variance], channel-concat",
    "b_var_avg": "(b) re-selected [variance], channel-averaged",
    "b_cov_concat": "(b) re-selected [cov], channel-concat",
    "b_cov_avg": "(b) re-selected [cov], channel-averaged",
    "b_meanpeak_concat": "(b) re-selected [mean-peak], channel-concat",
    "b_meanpeak_avg": "(b) re-selected [mean-peak], channel-averaged",
}


def write_markdown(out, width, fm_bins, band, selections, sel_full, stored,
                   results_full, results_test, verdicts, repro):
    L = []
    A = L.append
    A("# Audit: effect of the 255|256 channel seam on the real-data DFT feature\n")
    A(f"Generated {out['metadata']['date']} by "
      "`level1-features/experiments/audit_seam_dft.py`. Read-only on all existing files.\n")

    A("## Question\n")
    A("The DFT feature takes one 512-point rFFT per row. On real Open Iris codes the "
      "512 columns are two concatenated 256-angle phase channels, so that transform runs "
      "across the channel seam at 255|256 and its bins do not index true angular "
      "frequencies. The seam is measured, not assumed: mean adjacent-column bit agreement "
      "is 0.810 inside a channel but **0.328** across 255|256 (SIC-Gen, whose 512 columns "
      "are one true angular axis, shows no break: 0.836 vs 0.830).\n")

    A("## Where the per-channel band actually falls\n")
    ca = {c: selections[c]["channel_averaged"] for c in selections}
    A(f"- **(a) Frequency-matched:** `j_256 = k_512 / 2`, so the stored band 30-50 maps to "
      f"**bins {fm_bins[0]}-{fm_bins[-1]}** ({len(fm_bins)} bins).")
    A(f"- **(b) Stage 1's literal criterion (raw cross-code variance)** peaks at "
      f"**bin {ca['variance']['peak_bin']}** -> band {band['variance'][0]}-{band['variance'][-1]}.")
    A(f"- **(b) Normalised variance (variance/mean^2)** peaks at "
      f"**bin {ca['cov']['peak_bin']}** -> band {band['cov'][0]}-{band['cov'][-1]}.")
    A(f"- **(b) Mean-spectrum peak** is at **bin {ca['mean']['peak_bin']}** -> band "
      f"{band['mean'][0]}-{band['mean'][-1]}.")
    A(f"- Per-channel peaks agree across the two channels: variance "
      f"{selections['variance']['channel_0']['peak_bin']} / "
      f"{selections['variance']['channel_1']['peak_bin']}, mean "
      f"{selections['mean']['channel_0']['peak_bin']} / "
      f"{selections['mean']['channel_1']['peak_bin']}.")
    A(f"- For context, the same raw-variance procedure on the **512-point** spectra of "
      f"real codes peaks at bin **{sel_full['peak_bin']}**, against the stored band 30-50 "
      f"selected on synthetic data.\n")
    A("Raw variance is scale-dependent and spectral magnitude falls off with frequency, so "
      "on real codes it is dominated by the lowest bins. On the synthetic data Stage 1 was "
      "run on, this did not bite, because SIC-Gen has a genuine spectral peak where its "
      "mean and variance peaks coincide. The normalised and mean-peak criteria are "
      "reported because of that.\n")

    A("## Results — full protocol (1998 gallery, 16457 probes)\n")
    A("| Variant | Dim | R@10 | R@50 | R@100 | R@300 | Median rank |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    A(f"| **Stored 512-point (reference)** | {stored['dim']} | {stored['recall@10']:.4f} | "
      f"{stored['recall@50']:.4f} | {stored['recall@100']:.4f} | "
      f"{stored['recall@300']:.4f} | {stored['median_rank']:.1f} |")
    for name, lab in LABELS.items():
        r = results_full[name]
        A(f"| {lab} | {r['dim']} | {r['recall@10']:.4f} | {r['recall@50']:.4f} | "
          f"{r['recall@100']:.4f} | {r['recall@300']:.4f} | {r['median_rank']:.1f} |")

    A("\n### Delta vs stored, and effect size\n")
    A("| Variant | ΔR@10 | ΔR@50 | ΔR@100 | ΔR@300 | Δmedian rank | Effect |")
    A("|---|---:|---:|---:|---:|---:|---|")
    for name, lab in LABELS.items():
        if name == "baseline_512":
            continue
        d = verdicts[name]["delta_vs_stored_full_protocol"]
        A(f"| {lab} | {d['recall@10']:+.4f} | {d['recall@50']:+.4f} | "
          f"{d['recall@100']:+.4f} | {d['recall@300']:+.4f} | {d['median_rank']:+.1f} | "
          f"**{verdicts[name]['effect_size']}** |")
    A("\nEffect-size bands (max |ΔR@10|, |ΔR@50|): negligible < 0.005, small < 0.02, "
      "moderate < 0.05, material >= 0.05.\n")

    A("## Results — test-split probes only (clean read for (b))\n")
    A(f"Probes restricted to the {out['metadata']['n_probe_test_split']} probes of the 299 "
      "held-out test identities, ranked against the same full gallery. Variant (b)'s band "
      "was selected on train-split identities only, so this column is free of "
      "selection-on-test; it is shown for every variant so the comparison stays "
      "like-for-like. Absolute values are not comparable to the full-protocol table.\n")
    A("| Variant | Dim | R@10 | R@50 | R@100 | R@300 | Median rank |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, lab in LABELS.items():
        r = results_test[name]
        A(f"| {lab} | {r['dim']} | {r['recall@10']:.4f} | {r['recall@50']:.4f} | "
          f"{r['recall@100']:.4f} | {r['recall@300']:.4f} | {r['median_rank']:.1f} |")

    A("\n## Reproduction check\n")
    A(f"Recomputed 512-point baseline vs stored: `max |recall diff| = "
      f"{repro['max_abs_recall_diff']:.2e}` — **{repro['status']}**.\n")

    A("## Dimensions\n")
    A("| Variant | Bins kept per channel | Dimension |")
    A("|---|---:|---:|")
    A(f"| Stored / recomputed 512-point | 21 (of the 512-point transform) | 21 |")
    A(f"| Per-channel, channel-concatenated | {width} | {2 * width} |")
    A(f"| Per-channel, channel-averaged | {width} | {width} |")
    A("\nThe channel-averaged rows exist to separate the seam effect from the dimension "
      "effect: concatenation doubles the kept-bin count into the dimension, averaging does "
      "not.\n")
    _OUT_MD.write_text("\n".join(L) + "\n")


def make_plot(selections, sel_full, fm_bins, band):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    v = np.array(sel_full["variance_profile"])[1:]
    axes[0].plot(np.arange(1, len(v) + 1), v, color="#2a78d6", linewidth=1.4)
    axes[0].axvspan(30, 50, color="#d64545", alpha=0.18,
                    label="stored band 30-50 (selected on synthetic)")
    axes[0].axvline(sel_full["peak_bin"], color="#d64545", linestyle="--", linewidth=1.4,
                    label=f"real variance peak = bin {sel_full['peak_bin']}")
    axes[0].set_title("512-point transform, real train-split codes — cross-code variance")

    ca = selections["variance"]["channel_averaged"]
    for ax, key, title in (
        (axes[1], "variance_profile",
         "Per-channel 256-point — cross-code variance (Stage 1's criterion)"),
        (axes[2], "cov_profile",
         "Per-channel 256-point — normalised variance (variance / mean^2)"),
    ):
        for cname, col in (("channel_0", "#2a78d6"), ("channel_1", "#1baf7a")):
            v = np.array(selections["variance"][cname][key])[1:]
            ax.plot(np.arange(1, len(v) + 1), v, color=col, linewidth=1.4,
                    label=cname.replace("_", " "))
        ax.axvspan(fm_bins[0], fm_bins[-1], color="#d64545", alpha=0.18,
                   label=f"(a) freq-matched {fm_bins[0]}-{fm_bins[-1]}")
        crit = "variance" if key == "variance_profile" else "cov"
        ax.axvspan(band[crit][0], band[crit][-1], color="#e8a33d", alpha=0.25,
                   label=f"(b) {crit} band {band[crit][0]}-{band[crit][-1]}")
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Frequency bin (DC dropped)")
        ax.set_ylabel("Statistic across codes")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.01)

    plt.tight_layout()
    plt.savefig(str(_OUT_PNG), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
