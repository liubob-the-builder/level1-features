# Mask-geometry leakage audit, part 3 -- CircularConvEncoder v3 vs hand-designed features

Read-only, extends `audit_maskleak2_RESULTS.md`'s Check 5. Same 299-test-identity / 1998-gallery / 2488-de-duplicated-probe protocol and metrics as `audit_nn_v3_dedup_metrics.json` / `audit_all_features_dedup_testsplit.json`. Numbers only -- no conclusions beyond them.

**Strata verification:** the IoU quartile assignment used below was reconstructed (not stored as raw indices in `audit_maskleak2_results.json`) by re-running the identical deterministic computation, then verified to reproduce that file's stored per-quartile NN v3 numbers (n_probes, median rank, R@10/50/100/300) exactly, for all 4 quartiles, before anything below was computed. **PASSED.**

**Daugman/Hamming baseline:** not included. A plain IrisCode Hamming-distance identification baseline on this exact split is not already computed anywhere in this repo, and per instruction was skipped rather than built.

## Combined table -- quartile by quartile (Q1 = most similar masks, Q4 = most dissimilar)

| Feature | Quartile | n | R@10 | R@50 | R@100 | R@300 | MedRank |
|---|---|---|---|---|---|---|---|
| NN v3 | Q1 | 622 | 0.9502 | 0.9759 | 0.9904 | 0.9984 | 1.0 |
| NN v3 | Q2 | 622 | 0.8199 | 0.9180 | 0.9486 | 0.9839 | 1.0 |
| NN v3 | Q3 | 622 | 0.6752 | 0.8682 | 0.9196 | 0.9775 | 3.0 |
| NN v3 | Q4 | 622 | 0.5273 | 0.7588 | 0.8553 | 0.9325 | 9.0 |
| DFT | Q1 | 622 | 0.7122 | 0.8441 | 0.9019 | 0.9614 | 2.0 |
| DFT | Q2 | 622 | 0.4582 | 0.6833 | 0.7749 | 0.9148 | 15.0 |
| DFT | Q3 | 622 | 0.2958 | 0.5193 | 0.6479 | 0.8328 | 46.0 |
| DFT | Q4 | 622 | 0.1206 | 0.3119 | 0.4003 | 0.5997 | 181.5 |
| AC | Q1 | 622 | 0.6929 | 0.8183 | 0.8698 | 0.9373 | 2.0 |
| AC | Q2 | 622 | 0.3842 | 0.5643 | 0.6592 | 0.7990 | 30.5 |
| AC | Q3 | 622 | 0.2154 | 0.3633 | 0.4662 | 0.6785 | 119.5 |
| AC | Q4 | 622 | 0.1286 | 0.2540 | 0.3521 | 0.5370 | 248.0 |
| RL-C9 | Q1 | 622 | 0.6849 | 0.8232 | 0.8617 | 0.9132 | 2.0 |
| RL-C9 | Q2 | 622 | 0.4566 | 0.6270 | 0.7154 | 0.8650 | 16.0 |
| RL-C9 | Q3 | 622 | 0.2910 | 0.4952 | 0.6029 | 0.7830 | 52.0 |
| RL-C9 | Q4 | 622 | 0.1736 | 0.3441 | 0.4469 | 0.6463 | 135.0 |

## Q1-to-Q4 relative drop, per feature

Relative drop = (R@K[Q1] - R@K[Q4]) / R@K[Q1]. Median rank reported as raw Q1/Q4 values plus the Q4/Q1 ratio (higher rank = worse, so this is a increase factor, not a "drop").

| Feature | R@10 drop | R@50 drop | R@100 drop | R@300 drop | MedRank Q1 | MedRank Q4 | Q4/Q1 ratio |
|---|---|---|---|---|---|---|---|
| NN v3 | 0.4450 | 0.2224 | 0.1364 | 0.0660 | 1.0 | 9.0 | 9.00 |
| DFT | 0.8307 | 0.6305 | 0.5561 | 0.3763 | 2.0 | 181.5 | 90.75 |
| AC | 0.8144 | 0.6896 | 0.5952 | 0.4271 | 2.0 | 248.0 | 124.00 |
| RL-C9 | 0.7465 | 0.5820 | 0.4813 | 0.2923 | 2.0 | 135.0 | 67.50 |

## Aggregate (unstratified) hand-feature numbers, for reference

| Feature | R@10 | R@50 | R@100 | R@300 | MedRank | MeanRank |
|---|---|---|---|---|---|---|
| NN v3 | 0.7432 | 0.8802 | 0.9285 | 0.9731 | 2.0 | 35.23 |
| DFT | 0.3967 | 0.5896 | 0.6813 | 0.8272 | 26.0 | 169.93 |
| AC | 0.3553 | 0.5000 | 0.5868 | 0.7379 | 50.5 | 259.23 |
| RL-C9 | 0.4015 | 0.5723 | 0.6568 | 0.8018 | 28.0 | 195.52 |
