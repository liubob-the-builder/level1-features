# Embedding vs. hand-designed-feature representation comparison -- NN v3

Read-only audit. Same 299-test-identity / 1998-gallery / 2488-de-duplicated-probe protocol as `all_features_nn_testsplit_comparison.json` / `audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. Numbers only -- no interpretation beyond each result's own definition.

## Setup

- NN embeddings: quantized 64-dim (`CircularConvEncoder` v3), recomputed baseline verified to match `audit_nn_v3_dedup_metrics.json` (`quantized_deduplicated`) exactly.
- Hand-designed feature aggregate metrics recomputed here match `audit_all_features_dedup_testsplit.json` exactly for all four features (atol=1e-9): {'DFT': True, 'AC': True, 'RL-C9': True, 'RL-C6': True}
- **Part 1 sample set:** Union of gallery (1998 identities, stem '1') + de-duplicated probe (2488) vectors = 4486 total, each image used exactly once. Chosen (rather than probes-only or gallery-only) to maximize the sample-to-dimension ratio for the highest-dimensional feature (RL-C9, 576-dim): 4486 samples vs. 1998 (gallery-only, ratio ~3.5x) or 2488 (probes-only, ratio ~4.3x). Part 2 (error correlation) instead uses only the 2488-probe / 1998-gallery eval protocol, since it needs the actual identification ranks.
- **Part 2 sample set:** the 2488 de-duplicated test probes ranked against the 1998-identity gallery (identical to the matched-comparison benchmark).
- No sklearn in this environment; CCA and linear regression are hand-implemented (numpy/scipy only) -- see script docstring for exact method.
- CCA whitening uses NO explicit ridge regularization -- Cxx/Cyy are inverted via an eigenvalue pseudo-inverse with a relative cutoff (rcond=1e-10). Regression is plain OLS via `np.linalg.lstsq(rcond=None)`, also unregularized. Condition numbers are reported for every matrix inverted/whitened; a condition number > 1e+08 is flagged as near-singular.

## CCA self-checks (must pass before Part 1 CCA numbers are trusted)

| Feature | Self-pairing top corr (>= 0.999?) | Random-pairing top corr (< 0.5?) | Passed |
|---|---|---|---|
| DFT | 1.000000 (pass) | 0.1726 (pass) | PASSED |
| AC | 1.000000 (pass) | 0.2968 (pass) | PASSED |
| RL-C9 | 1.000000 (pass) | 0.4609 (pass) | PASSED |
| RL-C6 | 1.000000 (pass) | 0.4006 (pass) | PASSED |

All self-checks passed: **True**. Random-seed for the random-pairing self-check: 20260831.

## Part 1 -- Representation overlap (linear only)

| Feature | Dim | Top-3 canonical corr | cond(Cxx) | eff.rank(Cxx)/dim | cond(Cyy) | eff.rank(Cyy)/dim | Regression R^2 (mean) | cond(design) | design rank/dim |
|---|---|---|---|---|---|---|---|---|---|
| DFT | 21 | 0.8694, 0.5833, 0.3301 | 2.741e+01 | 21/21 | 3.153e+03 | 64/64 | 0.0793 | 6.986e+02 | 22/22 |
| AC | 160 | 0.9610, 0.9059, 0.8624 | 9.081e+03 | 160/160 | 3.153e+03 | 64/64 | 0.3393 | 9.070e+02 | 161/161 |
| RL-C9 | 576 | 0.9829, 0.9711, 0.9561 | 1.852e+04 | 576/576 | 3.153e+03 | 64/64 | 0.5353 | 2.349e+03 | 577/577 |
| RL-C6 | 384 | 0.9809, 0.9682, 0.9404 | 9.310e+03 | 384/384 | 3.153e+03 | 64/64 | 0.4783 | 2.395e+03 | 385/385 |

Regression R^2 range (min/max across the 64 NN dims), for reference:

| Feature | R^2 min | R^2 max |
|---|---|---|
| DFT | 0.0090 | 0.3842 |
| AC | 0.1423 | 0.6128 |
| RL-C9 | 0.3335 | 0.8846 |
| RL-C6 | 0.2671 | 0.8723 |

## Part 2 -- Error correlation (2488-probe eval protocol)

| Feature | Spearman rho | Spearman p | Jaccard@10 | Jaccard@50 | NN-fail-count@10 | Feature-fail-count@10 | NN rescues feature@10 | Feature rescues NN@10 |
|---|---|---|---|---|---|---|---|---|
| DFT | 0.4433 | 2.63e-120 | 0.3292 | 0.1980 | 639 | 1501 | 971 | 109 |
| AC | 0.4623 | 5.64e-132 | 0.3488 | 0.1880 | 639 | 1604 | 1024 | 59 |
| RL-C9 | 0.4661 | 2.12e-134 | 0.3537 | 0.2237 | 639 | 1489 | 933 | 83 |
| RL-C6 | 0.4724 | 1.73e-138 | 0.3537 | 0.2229 | 639 | 1493 | 936 | 82 |

| Feature | Jaccard@50 detail: NN-fail-count@50 | Feature-fail-count@50 |
|---|---|---|
| DFT | 298 | 1021 |
| AC | 298 | 1244 |
| RL-C9 | 298 | 1064 |
| RL-C6 | 298 | 1068 |
