# Do Gabor scale count and phase-channel layout explain the real-vs-synthetic bit gap?

`experiments/audit_singlescale_regions.py` | 2026-09-06 | commit `96b39de63d3a` | 1998 real gallery codes

Analysis only: no re-extraction, no retraining, no existing file modified. Every estimator is imported unmodified from `experiments/hd_distribution_real_vs_synthetic.py`.

## Verified code layout

In a real (32, 512) code: **rows 0-15 = Gabor scale 0** (coarse, `lambda_phi=28`), **rows 16-31 = Gabor scale 1** (fine, `lambda_phi=8`); **cols 0-255 = real phase channel**, **cols 256-511 = imaginary**, both over the same 256 angular positions. Confirmed from `iris/nodes/encoder/iris_encoder.py` and `extract_iris_code`, and empirically: adjacent-column agreement is ~0.88 everywhere except across 255|256 where it is 0.41, and agreement between column *j* and column *j*+256 is 0.5009 (quadrature, near-independent).

## Reproduction check

The full 32x512 region gives **621.8 DoF** and **mean run length 4.074** against the stored 621.8 / 4.074 -- **reproduced**. The slices below can be read against this baseline.

## All regions vs stored SIC-Gen

| Region | Bits | Impostor mean HD | Impostor sigma | DoF | Mean run length | Frac. runs 1-2 | Modal run | Mask valid |
|---|---|---|---|---|---|---|---|---|
| full (32x512) | 16384 | 0.4822 | 0.0200 | 621.8 | 4.074 | 0.387 | 2 | 0.676 |
| scale0_16x512 (16x512) | 8192 | 0.4715 | 0.0309 | 260.4 | 7.236 | 0.099 | 6 | 0.663 |
| scale0_16x256 (16x256) | 4096 | 0.4903 | 0.0319 | 245.0 | 7.007 | 0.096 | 7 | 0.651 |
| scale1_16x512 (16x512) | 8192 | 0.4923 | 0.0162 | 950.9 | 2.869 | 0.497 | 2 | 0.689 |
| scale1_16x256 (16x256) | 4096 | 0.4978 | 0.0180 | 769.5 | 2.993 | 0.421 | 3 | 0.690 |
| **SIC-Gen (stored, 32x512)** | 16384 | 0.4979 | 0.0330 | 229.0 | 6.010 | 0.097 | 6 | 0.940 |

Region key: `scale0` = rows 0-15 (coarse), `scale1` = rows 16-31 (fine); `16x512` = both phase channels, `16x256` = real channel only.

## (a) Does dropping to one scale (32x512 -> 16x512) move toward synthetic?

- **scale 0 (coarse):** DoF 621.8 -> 260.4 (toward synthetic); mean run length 4.074 -> 7.236 (toward synthetic, overshoots).
- **scale 1 (fine):** DoF 621.8 -> 950.9 (away from synthetic); mean run length 4.074 -> 2.869 (away from synthetic).

## (b) Does dropping the second phase channel (16x512 -> 16x256) move it further?

- **scale 0 (coarse):** DoF 260.4 -> 245.0 (toward synthetic); mean run length 7.236 -> 7.007 (toward synthetic).
- **scale 1 (fine):** DoF 950.9 -> 769.5 (toward synthetic); mean run length 2.869 -> 2.993 (toward synthetic).

## Does SIC-Gen match a single real scale, or sit between the two?

- **Degrees of freedom:** SIC-Gen at 229.0 against scale 0 = 260.4 and scale 1 = 950.9 at 16x512 -- **outside the range spanned by the two scales**. At 16x256 (scale 0 = 245.0, scale 1 = 769.5) it falls outside them. Closest single region: `scale0_16x256` (gap 16.0).
- **Mean run length:** SIC-Gen at 6.010 against scale 0 = 7.236 and scale 1 = 2.869 at 16x512 -- **between the two scales**. At 16x256 (scale 0 = 7.007, scale 1 = 2.993) it falls between them. Closest single region: `scale0_16x256` (gap 0.997).

## Caveats

- Each smaller region contains fewer bits (16384 -> 8192 -> 4096 valid-bit budget before masking), so per-pair HD is estimated from fewer samples and the impostor sigma widens partly for that reason alone -- some of the DoF movement across regions is a finite-sample effect, not a change in the underlying bit correlation.
- None of the real regions structurally matches SIC-Gen. A SIC-Gen code is a single continuous 512-angle axis with no phase-channel seam; the real 16x512 regions are 256 angles carried in two quadrature channels, and the real 16x256 regions are 256 angles in one channel. So even an exact numerical match would not mean the two codes have the same structure.
- The synthetic figures are read from the stored bit-level comparison and are not recomputed here; only the real regions were measured for this file.

