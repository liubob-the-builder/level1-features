# Fully-held-out gallery audit -- NN v3 vs hand-designed features

Read-only audit. Same 299-test-identity probe set (2488 de-duplicated probes) and same feature extraction/distances as `all_features_nn_testsplit_comparison.json` / `audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. The ONLY change is the gallery: 299 test identities only, instead of the full 1998 (train+val+test).

**Framing:** the 299-identity gallery has far fewer distractors than the 1998-identity gallery, so it is a strictly easier ranking task -- absolute recall rises for every feature at the smaller gallery size, for both the NN and the hand-designed features. The hand-designed features have no training exposure either way, so they benefit only from the smaller gallery; the NN benefits from the smaller gallery AND has no seen-in-training distractors. **The point of interest is therefore whether the NN-vs-hand-designed-feature gap in Recall@10 is preserved across the two gallery scales -- not the absolute recall values themselves.**

## NN vs hand-designed feature gap, Recall@10 (lead result)

| Feature | R@10 (1998-gallery) | R@10 (299-gallery) | Ratio NN/feature (1998) | Ratio NN/feature (299) | Abs. diff (1998) | Abs. diff (299) |
|---|---|---|---|---|---|---|
| **NN (v3, quantized)** | 0.7432 | 0.8814 | -- | -- | -- | -- |
| DFT | 0.3967 | 0.6117 | 1.873x | 1.441x | +0.3465 | +0.2697 |
| AC | 0.3553 | 0.5370 | 2.092x | 1.641x | +0.3879 | +0.3445 |
| RL-C9 | 0.4015 | 0.5969 | 1.851x | 1.477x | +0.3416 | +0.2846 |
| RL-C6 | 0.3999 | 0.5945 | 1.858x | 1.483x | +0.3432 | +0.2870 |

Ratio range across the 4 hand-designed features: 1.851x-2.092x at the 1998-gallery, 1.441x-1.641x at the 299-gallery. Absolute-difference range: +0.3416 to +0.3879 at the 1998-gallery, +0.2697 to +0.3445 at the 299-gallery.

## Full metrics, all features, both gallery scales

| Feature | Gallery | R@10 | R@50 | R@100 | R@300 | MedRank | MeanRank |
|---|---|---|---|---|---|---|---|
| NN (v3, quantized) | 1998 | 0.7432 | 0.8802 | 0.9285 | 0.9731 | 2.0 | 35.23 |
| NN (v3, quantized) | 299 | 0.8814 | 0.9703 | 0.9879 | 1.0000 | 1.0 | 6.99 |
| DFT | 1998 | 0.3967 | 0.5896 | 0.6813 | 0.8272 | 26.0 | 169.93 |
| DFT | 299 | 0.6117 | 0.8336 | 0.9184 | 1.0000 | 5.0 | 26.67 |
| AC | 1998 | 0.3553 | 0.5000 | 0.5868 | 0.7379 | 50.5 | 259.23 |
| AC | 299 | 0.5370 | 0.7468 | 0.8497 | 1.0000 | 8.0 | 41.31 |
| RL-C9 | 1998 | 0.4015 | 0.5723 | 0.6568 | 0.8018 | 28.0 | 195.52 |
| RL-C9 | 299 | 0.5969 | 0.8171 | 0.9064 | 1.0000 | 5.0 | 29.76 |
| RL-C6 | 1998 | 0.3999 | 0.5707 | 0.6612 | 0.8051 | 27.0 | 191.07 |
| RL-C6 | 299 | 0.5945 | 0.8151 | 0.9120 | 1.0000 | 5.0 | 29.13 |
| Christina (inverted) | 1998 | 0.4839 | 0.6797 | 0.7637 | 0.8830 | 12.0 | 121.40 |
| Christina (inverted) | 299 | 0.7030 | 0.8895 | 0.9477 | 1.0000 | 3.0 | 19.19 |

## Probe genuine-match coverage in the 299-identity gallery

| Feature | Probes | Missing genuine match | All present |
|---|---|---|---|
| NN (v3, quantized) | 2488 | 0 | True |
| DFT | 2488 | 0 | True |
| AC | 2488 | 0 | True |
| RL-C9 | 2488 | 0 | True |
| RL-C6 | 2488 | 0 | True |
| Christina (inverted) | 2488 | 0 | True |

## Methodology notes

- Probe set: identical 2488 de-duplicated probes (299 test identities, stems 2-10, excluding the 5 byte-identical duplicate probes) in every row above.
- Gallery restriction: for every feature, gallery vectors were filtered to the 299 test identities' own gallery image (stem "1"); no other change to extraction, distance, or ranking code.
- Every feature's 1998-gallery numbers shown above are the stored reference values (`audit_nn_v3_dedup_metrics.json` for the NN, `audit_all_features_dedup_testsplit.json` for the rest); this script independently recomputed each of them first and required an exact match (atol=1e-9) before trusting the corresponding 299-gallery number -- see `reference_checks` in the JSON output.
- Distances: SSD for NN/DFT/AC/RL-C9/RL-C6, L1 (`compute_l1_distance_plaintext`) for Christina's feature, matching the matched comparison exactly.