# Mask-geometry leakage audit, part 2 -- CircularConvEncoder v3

Read-only, extends `audit_maskleak_RESULTS.md`. Same 299-test-identity / 1998-gallery / 2488-de-duplicated-probe protocol and metrics as `audit_nn_v3_dedup_metrics.json`. Numbers only -- no conclusions beyond them.

## Reference baseline (unmodified NN v3, quantized)

| | R@10 | R@50 | R@100 | R@300 | MedRank |
|---|---|---|---|---|---|
| NN v3 (unmodified) | 0.7432 | 0.8802 | 0.9285 | 0.9731 | 2.0 |

## Check 4 -- constant-mask (all-ones) channel 1

Self-check (channel 1 = true mask reproduces `model(x)`): max abs diff = 0.000e+00 (< 1e-6 required).

| | R@10 | R@50 | R@100 | R@300 | MedRank |
|---|---|---|---|---|---|
| Constant-mask channel 1 | 0.5434 | 0.7199 | 0.7930 | 0.9180 | 7.0 |
| Baseline (unmodified) | 0.7432 | 0.8802 | 0.9285 | 0.9731 | 2.0 |

## Check 5 -- mask-dissimilarity stratification of genuine pairs

Gallery has exactly one image per identity (stem '1'), so every probe has exactly one genuine gallery match -- no multi-genuine-pair handling was needed. Genuine pairs (n=2488) split into 4 equal-sized quartiles of dissimilarity (Q1 = most similar masks, Q4 = most dissimilar). "Impostor dist" is each quartile's probes' mean per-probe-average SSD to all non-genuine gallery entries.

### Stratified by (1 - IoU) of the two true binary masks

| Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank | Mean genuine dist | Mean impostor dist |
|---|---|---|---|---|---|---|---|---|
| Q1 | 622 | 0.9502 | 0.9759 | 0.9904 | 0.9984 | 1.0 | 17757.56 | 102330.73 |
| Q2 | 622 | 0.8199 | 0.9180 | 0.9486 | 0.9839 | 1.0 | 26988.32 | 103010.69 |
| Q3 | 622 | 0.6752 | 0.8682 | 0.9196 | 0.9775 | 3.0 | 35080.66 | 106878.66 |
| Q4 | 622 | 0.5273 | 0.7588 | 0.8553 | 0.9325 | 9.0 | 46899.75 | 118980.20 |

### Stratified by 32-dim row-validity-profile Euclidean distance

| Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank | Mean genuine dist | Mean impostor dist |
|---|---|---|---|---|---|---|---|---|
| Q1 | 622 | 0.9469 | 0.9759 | 0.9887 | 1.0000 | 1.0 | 18692.39 | 104441.32 |
| Q2 | 622 | 0.8296 | 0.9341 | 0.9662 | 0.9823 | 1.0 | 28067.57 | 106956.33 |
| Q3 | 622 | 0.7074 | 0.8907 | 0.9341 | 0.9839 | 3.0 | 33793.64 | 107638.57 |
| Q4 | 622 | 0.4887 | 0.7203 | 0.8248 | 0.9260 | 11.5 | 46172.69 | 112164.04 |

## Check 6 -- partial correlation (extends Check 3)

n = 4968536 impostor pairs (same set as Check 3). Controlling for (v_A + v_B) and |v_A - v_B|, where v is each image's overall valid-bit fraction.

| | Pearson r | Spearman rho |
|---|---|---|
| Unconditional (Check 3 reference) | 0.3999 | 0.3813 |
| Partial, controlling v_sum & v_diff | 0.2650 | 0.2568 |

Partial Spearman computed by rank-transforming all four variables (embedding distance, mask-profile distance, v_sum, v_diff) and then taking the partial Pearson correlation of the ranks.
