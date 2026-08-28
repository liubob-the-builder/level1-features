# Level-1 Iris Filter Features (FHE-FIS)

Feature extraction and evaluation for Level-1 candidate filtering in the FHE-FIS
privacy-preserving iris identification pipeline. This work evaluates candidate
Level-1 iris-code features — a novel angular-DFT feature against the AC
(autocorrelation) and RL (run-length) baselines from Karakosta & Knottenbelt,
*"Iris Through the Looking Glass: Filter-Based Privacy-Preserving Identification"* —
ahead of encrypting the winning feature under FHE/BFV. Comparisons use SSD as the
distance metric in plaintext, mirroring the paper's encrypted Level-1 comparison operator.

Synthetic iris codes are produced by SIC-Gen (kept in the sibling `../sic-gen/`
directory), with local modifications recorded in `sicgen_modifications.diff`.

## Key findings

- **SIC-Gen substantially inflates Level-1 feature performance versus real data.**
  The paper's RL feature, implemented to its literal spec (C=6, 384-dim), scores
  Recall@50 = 77.6% on SIC-Gen versus the 38.3% reported on real CASIA-Iris-Thousand
  — roughly double. AC and DFT are similarly inflated (95% / 97% synthetic). Absolute
  synthetic numbers therefore do not predict real-data performance.

- **The gap is the data, not the implementation.** Because the ~2× inflation appears
  even on a faithful C=6 reproduction of the paper's own feature (and the C=9 variant
  differs by only ~4 points), the discrepancy cannot be attributed to implementation
  differences. All features approach saturation (95–100%) on synthetic data, so
  synthetic results should be read as *relative ordering only*, pending real-data validation.

- **Two candidate mechanisms for the inflation were tested and not supported.**
  (1) That DFT recovers SIC-Gen's per-subject mean run length — refuted: the DFT peak
  bin takes only 2–3 discrete values across 2000 subjects, far too coarse to drive the
  observed recall. (2) That the 32 rows are near-duplicates of one base row — weakened:
  PCA finds 21–24 effectively independent rows out of 32. The inflation is real but its
  exact mechanism remains unexplained; a real-CASIA comparison (via `row_redundancy_check.py`,
  built for this) is needed to resolve it.

- **DFT is competitive with AC and more compact.** On synthetic data DFT (21-dim) matches
  or slightly beats AC (160-dim) on recall/rank. Since DFT-magnitude and AC are Fourier
  pairs (both second-order), this is a compactness advantage rather than genuine
  complementarity — though their errors decorrelate on this data (Spearman ρ≈0.20,
  failure-set Jaccard 0.034).

- **DFT+AC is the strongest fusion.** RRF fusion of DFT+AC gives the largest gain
  (R@50 → 0.997) and beats the paper's AC+RL combination, at lower dimension. Adding RL
  as a third feature adds almost nothing once AC is present. (Gains are compressed by
  synthetic-data saturation and may not transfer in magnitude to real data.)

- **Open question — AC dimensionality.** The paper's stated 90-dim AC (90 = 5×18) does
  not follow from either aggregation mode implemented here (concat → 160-dim, mean → 5-dim).
  The construction of the 18 factor is unresolved and pending clarification from the authors.

## Repository structure

```
level1-features/
├── features/       feature extraction modules
├── experiments/    Stage 1/2/3 scripts and analyses
├── results/        saved ranks (.npz), metrics (.json), plots (.png)
├── sicgen_modifications.diff   local edits to the SIC-Gen generator
├── requirements.txt
└── README.md
```

Galleries live in `../sic-gen/` (the SIC-Gen generator produces them there) and are
not tracked in this repo; they are regenerable from the recorded seed + process count
(see each gallery's `metadata.json`).

## Local modification to the generator

`downsampling_every_n_row` / `downsampling_every_n_column` in `sic-gen.py` were changed
from the upstream `(8, 2)` to `(2, 0)`, so codes are 32×512 instead of the upstream
default resolution. A `--seed` / `-s` CLI argument was also added (not upstream): it
seeds both `numpy.random` and stdlib `random` (the latter because `template.py`'s
`flip_edge()` draws from it), with per-worker seed = `seed + first_subject_number_in_chunk`.
A gallery is therefore reproducible only for a fixed `(seed, -p)` pair, since a different
process count re-partitions subjects into different chunks. The exact diff is captured in
`sicgen_modifications.diff`.

## Feature extraction modules (`features/`)

| File | What it does |
|---|---|
| `dft_spectrum.py` | The DFT feature used in Stage 1/2/3: per-row real FFT magnitude spectrum along the circular 512-column axis, aggregated across rows (`average` or `concat`). The final feature keeps bins 30–50 (21-dim). |
| `dft_features.py` | Earlier, exploratory DFT feature (column-density profile → power spectrum → 16 log-spaced bins). Superseded by `dft_spectrum.py`; kept for reference. |
| `ac_features.py` | Autocorrelation lag-energy feature (paper's AC baseline): 5 lag-range energy bins per row, in `concat` (5×32=160-dim, used in Stage 2/3) or `mean` (validity-weighted across rows, 5-dim, weaker) mode. The paper's stated 90-dim follows from neither mode — see Open question above. |
| `rl_features.py` | Run-length histogram feature (paper's RL baseline): per-row histogram of 0-run and 1-run lengths, `2*C` bins/row × 32 rows. `C=9` (576-dim) matches the paper's stated dimension; `C=6` (384-dim) matches its literal "C=6" description — both evaluated side by side. |

## Learned Level-1 feature (`features/`, neural network alternative)

A learned, mask-aware, BFV-compatible alternative to the hand-crafted Level-1 features above: a small circularly-padded CNN (`CircularConvEncoder`) trained with a batch-hard SSD triplet loss on real CASIA-Iris-Thousand codes (`casia-extraction/casia-codes-2d/`), with an exact-rotation-invariance guarantee built into the architecture rather than the input. All code for this feature lives in `features/`, including its training driver, per explicit instruction (elsewhere in this repo, run scripts live in `experiments/`).

| File | What it does |
|---|---|
| `nn_split.py` | Subject-level train/val/test split of the 1998 usable identities (excludes `524_L`/`668_L`, which have no gallery image). Both eyes of a subject always share a split. |
| `nn_dataset.py` | Template/mask loading, input-tensor construction, and the `PairBatchDataset` batch sampler (distinct identities per batch, for in-batch hard-negative mining) plus embedding helpers used at eval time. |
| `nn_model.py` | `CircularConvEncoder` (circular-padded conv stack + masked circular pooling + tanh-bounded embedding + integer quantization) and the SSD-based loss functions (`SSDTripletLoss`, `BatchHardSSDTripletLoss`). |
| `nn_train.py` | Training driver: builds/loads the split, trains `CircularConvEncoder`, evaluates Recall@K/rank each epoch against a train+val gallery pool, checkpoints the best epoch, and optionally runs a final held-out test evaluation against the full 1998-identity gallery pool. |
| `nn_eval_quantized.py` | Loads a trained checkpoint and compares Recall@K/rank between its raw float embeddings and its BFV-compatible quantized (integer) embeddings, on the same held-out test protocol as `nn_train.py --final-eval`. |

## Experiments (`experiments/`)

### Stage 1 — DFT bin selection (300-subject gallery)
| File | What it does |
|---|---|
| `stage1_spectrum_inspection.py` | Mean + per-bin-variance angular DFT magnitude spectra across the 300-subject gallery, to choose which frequency bins carry signal. **Finding: mean/variance both peak at bins ~38–41.** |
| `stage1_signal_check.py` | Genuine-vs-impostor SSD separation (mean/std/d′, histograms) for candidate bin sets. **Result: bins 30–50 chosen (d′≈1.64); the 78–82 harmonic added negligible separation and was dropped.** |

### Stage 2 — full head-to-head evaluation (2000-subject gallery)
| File | What it does |
|---|---|
| `stage2_evaluate.py` | Main evaluation: extracts DFT / AC / RL-C9 / RL-C6 for gallery (IC1) and probe (IC2), computes SSD-based ranks, reports Recall@10/50/100/300, median/mean rank, EER, d′, dimension per feature — plus Spearman rank correlation and failure-set Jaccard overlap between every pair. |
| `stage2_run_length_artifact_check.py` | Negative control: tests whether DFT's performance is an artifact of SIC-Gen encoding identity via per-subject mean run length. **Result: the tested mechanism is refuted — the DFT peak bin is too coarse (2–3 values across 2000 subjects) to explain the recall; effects were in the predicted direction but far too weak.** |
| `row_redundancy_check.py` | Reusable diagnostic (not paper-specific): quantifies how redundant a code's 32 rows are — bit-domain Hamming distance (unaligned and aligned ±7), spectral correlation, and PCA effective-dimensionality in both domains. Built to run identically against any template directory (SIC-Gen folders or a flat CASIA-style export) for later real-vs-synthetic comparison. **Result: 21–24 effective rows out of 32.** |

### Stage 3 — Reciprocal Rank Fusion
| File | What it does |
|---|---|
| `stage3_fusion.py` | Fuses DFT / AC / RL-C9 pairwise and all three via RRF (k=60) on top of Stage 2's ranks, re-ranking each probe. Reports the recall/rank table per combination plus each fusion's R@50 gain over its best single component. **Result: DFT+AC best (R@50 0.997); adding RL barely helps once AC is present.** |

### Ad hoc validation
| File | What it does |
|---|---|
| `validate_gen.py` | Standalone genuine/impostor fractional-HD check (aligned ±*r* vs unaligned) for a generated directory; two-panel histogram. General sanity check, not part of Stage 1/2/3. |
| `run_christina_rl.py` | Runs Christina's enhanced-RL feature (`compute_enhanced_run_stats`/`compute_l1_distance_plaintext` from `christina-fhe-fis/filter_fhe_iris_complete.py`, imported read-only) on the 300-subject gallery under both SIC-Gen mask conventions (raw vs. inverted), reporting Recall@K/median/mean rank/dimension/dropped-row-% side by side via her own `compute_recall_at_k`. |
| `run_christina_rl_full2000.py` | Extends `run_christina_rl.py` to the full 2000-subject SIC-Gen gallery (no sampling), same single-scale `(1,32,512)` code shape and both mask conventions, for direct comparison against the 300-subject run at full Stage-2 scale. |
| `run_christina_rl_casia.py` | Real-CASIA replication of `run_christina_rl.py`: runs her feature (unmodified, `(32,512)` codes fed with no reshaping, reproducing her real 32-pseudo-scale behavior) on 300 real CASIA-Iris-Thousand identities from `casia-extraction/casia-codes-christina/` under both mask conventions, and compares Recall@50 against the paper's reported real-CASIA RL figure. |
| `run_christina_rl_casia_full2000.py` | Re-runs `run_christina_rl_casia.py` at full 1000-subject/2000-identity scale (1998 usable galleries, excluding `524_L`/`668_L`; 16,457 probes), under both mask conventions, comparing Recall@50 against the paper's reported real-CASIA RL figure (38.3%) and against the other full2000 evaluations. |
| `eval_myfeatures_realcasia.py` | Real-CASIA validation of Stage 2: runs DFT/AC/RL-C9/RL-C6 (same extraction calls as `stage2_evaluate.py`, generalized to real CASIA's 1-gallery/3-9-probes-per-identity structure) on `casia-extraction/casia-codes-2d/`, reporting Recall@K/median/mean rank/EER/d′ and Spearman/Jaccard error-correlation side by side with the synthetic Stage 2 numbers, plus mask-handling notes per feature (DFT/RL are mask-blind, AC is mask-aware). |
| `eval_myfeatures_realcasia_maskaware.py` | Re-runs the real-CASIA evaluation with DFT and RL-C9/RL-C6 now using their new optional `mask` parameter (row-dropping + mean-fill for DFT, valid-segment-only run-lengths for RL; AC unchanged, was already mask-aware), reporting a three-way real-blind / real-aware / synthetic comparison table plus mask-aware Spearman/Jaccard error-correlation, against `myfeatures_realcasia.json`. |
| `eval_myfeatures_synthetic_maskaware.py` | Fills the fourth 2x2 cell: runs mask-aware DFT/RL-C9/RL-C6 on the 2000-subject synthetic gallery (AC excluded, unchanged by construction), comparing against `stage2_results.json`'s mask-blind baseline; confirms and reports the SIC-Gen mask convention (1=valid) and mean valid-fraction (~94%) versus real CASIA's (~68%). |
| `eval_fusion_maskaware.py` | Re-runs Stage 3's RRF fusion (DFT+AC, DFT+RL-C9, AC+RL-C9, DFT+AC+RL-C9, k=60) with mask-aware features on both the synthetic 2000-subject gallery and real CASIA `casia-codes-2d/`, reporting each combo's R@10/50/100/MedRank plus its R@50 gain over the best single component, and mask-aware Spearman/Jaccard error-correlation, on both datasets side by side. **Result: fusion gain is larger on real data (+0.111 R@50, DFT+AC+RL-C9) than synthetic (+0.018), opposite of the naive "complementarity collapses on real data" expectation — synthetic's ceiling effect (DFT alone already at R@50=0.997) leaves little room for fusion to help, regardless of correlation.** |
| `eval_myfeatures_realcasia_full2000.py` | Re-runs the mask-aware real-CASIA single-feature evaluation (DFT/AC/RL-C9/RL-C6) at full 1000-subject/2000-identity scale (1998 usable galleries, excluding `524_L`/`668_L` for a failed gallery image), removing the 300-identity gallery-size confound; compares directly against the N=2000 synthetic mask-aware numbers. |
| `eval_fusion_realcasia_full2000.py` | Re-runs `eval_fusion_maskaware.py`'s real-CASIA RRF fusion side at full 1998-gallery scale (synthetic side reused unchanged, already N=2000), reporting R@10 fusion gain over the best single component and mask-aware Spearman/Jaccard error-correlation alongside the existing N=2000 synthetic fusion numbers. |
| `eval_myfeatures_realcasia_full2000_maskblind.py` | Mask-blind counterpart to `eval_myfeatures_realcasia_full2000.py` (DFT/RL-C9/RL-C6 called with `mask=None`, the default mask-blind path; AC skipped since it's inherently mask-aware), at the same full 1998-gallery/16,457-probe scale, for a blind-vs-aware comparison against that file's mask-aware results. |
| `eval_all_features_nn_testsplit.py` | Head-to-head comparison of all Level-1 features (mask-aware DFT/AC/RL-C9/RL-C6, Christina's enhanced-RL under both mask conventions, and the learned NN v3) on the identical held-out split used for the NN's final test evaluation: probes = only the 299 `nn_identity_split.json` test identities, gallery = the full 1998-identity pool, matching `nn_train.py --final-eval`'s protocol exactly for a fair apples-to-apples comparison. |
| `audit_nn_v3_verification.py` | Read-only adversarial audit of the NN v3 result: split disjointness, gallery/probe construction, quantization consistency, distance/ranking correctness (incl. manual per-probe spot-checks), a random-embedding chance-floor baseline, rank-distribution recomputation, and monotonicity/BatchNorm/dev-pool checks — all independently re-derived from the real checkpoint rather than trusting stored summaries. |
| `audit_shuffle_test.py` | Label-shuffle leakage detector: trains a throwaway model (never touches any real checkpoint) with the probe↔gallery identity correspondence derangement-scrambled within every batch, then evaluates it on the real test protocol to confirm recall collapses toward chance. |

## Results (`results/`)

Saved outputs backing every table and figure. Rank arrays (`stage2_ranks.npz`,
`stage3_fusion_ranks.npz`) are the primary artifacts — every metric and plot can be
recomputed from them without regenerating any iris codes. Also includes the metric
JSONs, the negative-control and row-redundancy data, and the plots.

## Reproducing

- Galleries: regenerable from `../sic-gen` with the recorded seed and process count
  (see each gallery's `metadata.json`), plus the edits in `sicgen_modifications.diff`.
- Environment: `pip install -r requirements.txt`.
- Run scripts from the repository root (paths are anchored via `Path(__file__)`, so
  they resolve regardless of working directory).
