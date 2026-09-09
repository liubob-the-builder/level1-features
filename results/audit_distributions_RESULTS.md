# Genuine / impostor distance distributions -- learned embedding vs Christina's RL

Date: 2026-09-07  |  git commit: `96b39de63d3a079f9b51b65267d0d431a26aceb5`

Full genuine and impostor distance distributions, emitted as fine-resolution histograms so separation can be plotted directly instead of being summarised only by d' and means. Histogram data lives in `level1-features/results/audit_distributions_results.json`.

## Protocol

- Probes: 299 held-out test identities, stems 2-10, de-duplicated to **2488 probes** (excluded: 029_L_3, 230_R_5, 236_R_2, 543_L_2, 703_L_7 -- byte-identical to their own gallery image (CASIA-Iris-Thousand source-data artifact, see audit_nn_v3_verification.json check_2).).
- Gallery: **1998 identities**, stem `1` (train+val+test; `524_L`/`668_L` already absent from the split).
- Genuine pairs: distance from each probe to its OWN identity's single gallery entry (stem '1') -> one per probe, 2488 total.
- Impostor pairs: distance from each probe to every OTHER identity's gallery entry -> 2488 * 1997 = 4968536 total.
- Sanity gate: ranks re-derived from these exact distance matrices reproduce the published de-duplicated numbers to `atol=1e-9`.

## :warning: The two features are NOT on a common axis

The two features use DIFFERENT distance metrics (SSD vs weighted L1) and DIFFERENT value ranges. Their histograms are NOT on a common axis and their raw distance values are NOT directly comparable. Each feature's histograms use its own bin range; genuine and impostor share bin edges only WITHIN a feature. No normalisation onto a shared axis has been applied. Compare the features via d'/EER/recall, never by overlaying their raw distance histograms.

## Metric and value range per feature

| Feature | Metric | Dim | Genuine range | Impostor range | Shared bin range (plot axis) |
|---|---|---|---|---|---|
| NN (v3, learned, quantized) | SSD | 64 | [3720, 1.468e+05] | [9967, 4.486e+05] | [3720, 4.486e+05] |
| Christina RL (inverted) | L1 (weighted) | 576 | [150.6, 353.3] | [166.2, 670.4] | [150.6, 670.4] |

## Separation summary

| Feature | Genuine mean +/- std | Impostor mean +/- std | d' | EER | median rank | R@10 |
|---|---|---|---|---|---|---|
| NN (v3, learned, quantized) | 3.168e+04 +/- 1.852e+04 | 1.078e+05 +/- 4.088e+04 | 2.3985 | 0.0820 | 2.0 | 0.7432 |
| Christina RL (inverted) | 222.8 +/- 25.8 | 281 +/- 39.14 | 1.7553 | 0.1612 | 12.0 | 0.4839 |

d' and EER are computed on the same genuine/impostor sets the histograms describe, using the unmodified `stage2_evaluate.d_prime` / `stage2_evaluate.eer`. Because the metrics differ, d' and EER (both scale-free) are the valid cross-feature comparison; the raw distance means are not.

## Metric detail

- **NN (v3, learned, quantized)** -- Sum of squared differences (nn_train.ssd_matrix) on the BFV-compatible QUANTIZED integer embedding produced by CircularConvEncoder.quantize (quant_range=127, embedding_dim=64), checkpoint nn_level1_v3_checkpoint_best.pt.
- **Christina RL (inverted)** -- Weighted L1 via Christina's unmodified compute_l1_distance_plaintext with FilterFHEConfig() defaults: 0.50*hist(scale_0) + 0.20*sum(hist(scale_1..31)) + 0.20*mean(row_diffs) + 0.10*mean(spatial_diffs), aggregated over 32 scales. This is the metric the matched comparison ranked with -- NOT SSD. Mask convention: inverted/fixed (occluded_mask = 1 - native_mask).

## How to plot

For each feature, one panel: `histograms.shared_edges.bin_edges` (512 bins) with `genuine_counts` and `impostor_counts` overlaid. `histograms.own_range` gives each set re-binned over its own min..max (256 bins) for shape detail -- those two must not be overlaid on each other. Never place the two features in the same panel or on a shared axis.
