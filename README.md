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
| `run_christina_rl_casia.py` | Real-CASIA replication of `run_christina_rl.py`: runs her feature (unmodified, `(32,512)` codes fed with no reshaping, reproducing her real 32-pseudo-scale behavior) on 300 real CASIA-Iris-Thousand identities from `casia-extraction/casia-codes-christina/` under both mask conventions, and compares Recall@50 against the paper's reported real-CASIA RL figure. |
| `eval_myfeatures_realcasia.py` | Real-CASIA validation of Stage 2: runs DFT/AC/RL-C9/RL-C6 (same extraction calls as `stage2_evaluate.py`, generalized to real CASIA's 1-gallery/3-9-probes-per-identity structure) on `casia-extraction/casia-codes-2d/`, reporting Recall@K/median/mean rank/EER/d′ and Spearman/Jaccard error-correlation side by side with the synthetic Stage 2 numbers, plus mask-handling notes per feature (DFT/RL are mask-blind, AC is mask-aware). |
| `eval_myfeatures_realcasia_maskaware.py` | Re-runs the real-CASIA evaluation with DFT and RL-C9/RL-C6 now using their new optional `mask` parameter (row-dropping + mean-fill for DFT, valid-segment-only run-lengths for RL; AC unchanged, was already mask-aware), reporting a three-way real-blind / real-aware / synthetic comparison table plus mask-aware Spearman/Jaccard error-correlation, against `myfeatures_realcasia.json`. |
| `eval_myfeatures_synthetic_maskaware.py` | Fills the fourth 2x2 cell: runs mask-aware DFT/RL-C9/RL-C6 on the 2000-subject synthetic gallery (AC excluded, unchanged by construction), comparing against `stage2_results.json`'s mask-blind baseline; confirms and reports the SIC-Gen mask convention (1=valid) and mean valid-fraction (~94%) versus real CASIA's (~68%). |
| `eval_fusion_maskaware.py` | Re-runs Stage 3's RRF fusion (DFT+AC, DFT+RL-C9, AC+RL-C9, DFT+AC+RL-C9, k=60) with mask-aware features on both the synthetic 2000-subject gallery and real CASIA `casia-codes-2d/`, reporting each combo's R@10/50/100/MedRank plus its R@50 gain over the best single component, and mask-aware Spearman/Jaccard error-correlation, on both datasets side by side. **Result: fusion gain is larger on real data (+0.111 R@50, DFT+AC+RL-C9) than synthetic (+0.018), opposite of the naive "complementarity collapses on real data" expectation — synthetic's ceiling effect (DFT alone already at R@50=0.997) leaves little room for fusion to help, regardless of correlation.** |

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
