# Audit: effect of the 255|256 channel seam on the real-data DFT feature

Generated 2026-09-03 by `level1-features/experiments/audit_seam_dft.py`. Read-only on all existing files.

## Question

The DFT feature takes one 512-point rFFT per row. On real Open Iris codes the 512 columns are two concatenated 256-angle phase channels, so that transform runs across the channel seam at 255|256 and its bins do not index true angular frequencies. The seam is measured, not assumed: mean adjacent-column bit agreement is 0.810 inside a channel but **0.328** across 255|256 (SIC-Gen, whose 512 columns are one true angular axis, shows no break: 0.836 vs 0.830).

## Where the per-channel band actually falls

- **(a) Frequency-matched:** `j_256 = k_512 / 2`, so the stored band 30-50 maps to **bins 15-25** (11 bins).
- **(b) Stage 1's literal criterion (raw cross-code variance)** peaks at **bin 1** -> band 1-11.
- **(b) Normalised variance (variance/mean^2)** peaks at **bin 1** -> band 1-11.
- **(b) Mean-spectrum peak** is at **bin 17** -> band 12-22.
- Per-channel peaks agree across the two channels: variance 17 / 1, mean 17 / 1.
- For context, the same raw-variance procedure on the **512-point** spectra of real codes peaks at bin **2**, against the stored band 30-50 selected on synthetic data.

Raw variance is scale-dependent and spectral magnitude falls off with frequency, so on real codes it is dominated by the lowest bins. On the synthetic data Stage 1 was run on, this did not bite, because SIC-Gen has a genuine spectral peak where its mean and variance peaks coincide. The normalised and mean-peak criteria are reported because of that.

## Results — full protocol (1998 gallery, 16457 probes)

| Variant | Dim | R@10 | R@50 | R@100 | R@300 | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| **Stored 512-point (reference)** | 21 | 0.3707 | 0.5586 | 0.6538 | 0.8053 | 31.0 |
| Recomputed 512-point (self-check) | 21 | 0.3707 | 0.5586 | 0.6538 | 0.8053 | 31.0 |
| (a) freq-matched 15-25, channel-concat | 22 | 0.2936 | 0.4970 | 0.5983 | 0.7734 | 52.0 |
| (a) freq-matched 15-25, channel-averaged | 11 | 0.2717 | 0.4813 | 0.5872 | 0.7675 | 57.0 |
| (b) re-selected [variance], channel-concat | 22 | 0.3115 | 0.4732 | 0.5653 | 0.7310 | 63.0 |
| (b) re-selected [variance], channel-averaged | 11 | 0.2766 | 0.4554 | 0.5558 | 0.7394 | 69.0 |
| (b) re-selected [cov], channel-concat | 22 | 0.3115 | 0.4732 | 0.5653 | 0.7310 | 63.0 |
| (b) re-selected [cov], channel-averaged | 11 | 0.2766 | 0.4554 | 0.5558 | 0.7394 | 69.0 |
| (b) re-selected [mean-peak], channel-concat | 22 | 0.3133 | 0.5091 | 0.6125 | 0.7845 | 47.0 |
| (b) re-selected [mean-peak], channel-averaged | 11 | 0.2910 | 0.4934 | 0.5998 | 0.7777 | 53.0 |

### Delta vs stored, and effect size

| Variant | ΔR@10 | ΔR@50 | ΔR@100 | ΔR@300 | Δmedian rank | Effect |
|---|---:|---:|---:|---:|---:|---|
| (a) freq-matched 15-25, channel-concat | -0.0770 | -0.0616 | -0.0555 | -0.0319 | +21.0 | **material** |
| (a) freq-matched 15-25, channel-averaged | -0.0989 | -0.0773 | -0.0665 | -0.0378 | +26.0 | **material** |
| (b) re-selected [variance], channel-concat | -0.0592 | -0.0854 | -0.0885 | -0.0743 | +32.0 | **material** |
| (b) re-selected [variance], channel-averaged | -0.0941 | -0.1032 | -0.0980 | -0.0659 | +38.0 | **material** |
| (b) re-selected [cov], channel-concat | -0.0592 | -0.0854 | -0.0885 | -0.0743 | +32.0 | **material** |
| (b) re-selected [cov], channel-averaged | -0.0941 | -0.1032 | -0.0980 | -0.0659 | +38.0 | **material** |
| (b) re-selected [mean-peak], channel-concat | -0.0574 | -0.0495 | -0.0413 | -0.0208 | +16.0 | **material** |
| (b) re-selected [mean-peak], channel-averaged | -0.0797 | -0.0652 | -0.0540 | -0.0276 | +22.0 | **material** |

Effect-size bands (max |ΔR@10|, |ΔR@50|): negligible < 0.005, small < 0.02, moderate < 0.05, material >= 0.05.

## Results — test-split probes only (clean read for (b))

Probes restricted to the 2493 probes of the 299 held-out test identities, ranked against the same full gallery. Variant (b)'s band was selected on train-split identities only, so this column is free of selection-on-test; it is shown for every variant so the comparison stays like-for-like. Absolute values are not comparable to the full-protocol table.

| Variant | Dim | R@10 | R@50 | R@100 | R@300 | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Recomputed 512-point (self-check) | 21 | 0.3979 | 0.5905 | 0.6819 | 0.8275 | 26.0 |
| (a) freq-matched 15-25, channel-concat | 22 | 0.3169 | 0.5146 | 0.6101 | 0.7850 | 45.0 |
| (a) freq-matched 15-25, channel-averaged | 11 | 0.2936 | 0.4978 | 0.6017 | 0.7830 | 52.0 |
| (b) re-selected [variance], channel-concat | 22 | 0.3414 | 0.4902 | 0.5800 | 0.7304 | 54.0 |
| (b) re-selected [variance], channel-averaged | 11 | 0.3077 | 0.4705 | 0.5672 | 0.7401 | 61.0 |
| (b) re-selected [cov], channel-concat | 22 | 0.3414 | 0.4902 | 0.5800 | 0.7304 | 54.0 |
| (b) re-selected [cov], channel-averaged | 11 | 0.3077 | 0.4705 | 0.5672 | 0.7401 | 61.0 |
| (b) re-selected [mean-peak], channel-concat | 22 | 0.3365 | 0.5371 | 0.6354 | 0.7982 | 39.0 |
| (b) re-selected [mean-peak], channel-averaged | 11 | 0.3137 | 0.5187 | 0.6245 | 0.7898 | 45.0 |

## Reproduction check

Recomputed 512-point baseline vs stored: `max |recall diff| = 0.00e+00` — **reproduced**.

## Dimensions

| Variant | Bins kept per channel | Dimension |
|---|---:|---:|
| Stored / recomputed 512-point | 21 (of the 512-point transform) | 21 |
| Per-channel, channel-concatenated | 11 | 22 |
| Per-channel, channel-averaged | 11 | 11 |

The channel-averaged rows exist to separate the seam effect from the dimension effect: concatenation doubles the kept-bin count into the dimension, averaging does not.

