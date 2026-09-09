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

## Setup

### Python version


Two environments were used, and they are **not** interchangeable:

| Environment | Python | Used for | Evidence |
|---|---|---|---|
| Analysis — this repo | **3.12.3** (`../.venv`) | everything in `features/` and `experiments/` | the only environment with `torch` and `scipy`; 25 of the 28 `__pycache__` files are `cpython-312`, including `nn_train`, `nn_model`, `nn_dataset` and every `audit_*` |
| Extraction — `../casia-extraction/` | **3.11.15** (conda env `iris`) | producing the CASIA codes with `open-iris` | the only environment with `open_iris` 1.11.1; every traceback in `results/casia_extraction_manifest.json` is under `python3.11`. `extract_casia.py`'s own docstring says "Run in the `iris` conda env (has open-iris + numpy 1.24.4; the project .venv does not)" |

The extraction environment has neither `torch` nor `scipy`, so it cannot run this repo's NN
or audit scripts; the analysis environment has no `open-iris`, so it cannot run extraction.

### Dependencies

```bash
pip install -r requirements.txt
```

Pinned to the analysis environment: `numpy` 2.4.4, `scipy` 1.18.0, `matplotlib` 3.10.8,
`torch` 2.13.0 (CPU wheel), `pillow` 12.1.1. Training and every audit script need `torch`;
`pillow` is needed only by `render_casia_iris_code_image.py`.

### Sibling-directory dependencies

Three directories outside this repo are required. They are not submodules and not tracked
here — the layout is assumed by `Path(__file__)`-relative constants in the scripts, so the
repo must sit alongside them:

```
iris-recognition/
├── level1-features/     <- this repo
├── sic-gen/             <- SIC-Gen generator + the synthetic galleries
├── casia-extraction/    <- real CASIA codes and the extraction script
└── christina-fhe-fis/   <- Christina's FHE-FIS reference implementation
```

- **`../sic-gen/`** — supplies the galleries `sicgen_gallery_300subjects/` (Stage 1) and
  `sicgen_gallery_2000subjects/` (Stage 2/3), and the `template` module imported by
  `experiments/validate_gen.py`. Without it, every synthetic-data script fails.
- **`../casia-extraction/`** — supplies `casia-codes-2d/` (used by the DFT/AC/RL features
  and the NN) and `casia-codes-christina/` (used by Christina's enhanced-RL feature), plus
  `extract_casia.py`, which produced both. Referenced by 25 path constants across the
  scripts. Without it, every real-data script fails. See "Data acquisition" below.
- **`../christina-fhe-fis/`** — read-only reference implementation. Nine scripts do
  `from filter_fhe_iris_complete import ...` (`FilterFHEConfig`,
  `compute_enhanced_run_stats`, `compute_l1_distance_plaintext`, `extract_iris_code`,
  `extract_feature_vector_all_scales`) and `run_christina_rl_casia.py` additionally does
  `from evaluation import compute_recall_at_k`. Without it those scripts raise `ImportError`
  at import time, before doing any work.

Note on the warnings her module prints on import: `filter_fhe_iris_complete.py` emits
`⚠️  TenSEAL not available - using simulation mode` and `⚠️  open-iris not available` when
those optional packages are missing, and `evaluation.py` emits `⚠️  System import failed: ...`.
These are **expected** in the analysis environment and are not errors — the plaintext
feature/distance path this repo uses still runs, as recorded in
`experiments/christina_full2000.log`, which is the log of a complete successful full-2000 run
that printed all three.

### Data acquisition

**Synthetic (SIC-Gen) results are fully reproducible from this repo plus `../sic-gen/`.**
Galleries are regenerable from the recorded seed and process count (see each gallery's
`metadata.json`) with the edits in `sicgen_modifications.diff` applied.

**Real-data results are not**, because they depend on an externally-obtained,
licence-restricted dataset and on extraction code that lives outside this repo:

- **CASIA-Iris-Thousand** is a third-party dataset. It is not redistributed here, in any form,
  and must be obtained separately.
- Extraction runs **`open-iris` version 1.11.1** (recorded as `open_iris_version` in
  `results/casia_extraction_manifest.json`, and matching the `open_iris-1.11.1` package in
  the `iris` conda environment) under Python 3.11.
- The extraction script itself, `casia-extraction/extract_casia.py`, is **not in this repo**.
  This repo consumes only its outputs (`casia-codes-2d/`, `casia-codes-christina/`).

So a third party can reproduce the whole synthetic side unaided, and can reproduce the real
side only after obtaining CASIA and re-running extraction.

## Running

All scripts anchor their paths via `Path(__file__)`, so they run from any working directory;
the commands below are written from the repository root. Most scripts take no arguments —
where a script has a command-line interface, its flags are shown.

### 1. Extraction (outside this repo)

Produces the real CASIA codes that every real-data script reads. Run in the Python 3.11
`iris` environment, from the `casia-extraction/` directory — not from this repo:

```bash
# in ../casia-extraction/, Python 3.11 env with open-iris 1.11.1
python extract_casia.py --first-n 1000 --run-label full-1000
```

`extract_casia.py` requires exactly one of `--subjects <IDs...>` or `--first-n <N>`, and
takes an optional `--run-label` (default `pilot`) that is recorded in its `manifest.json`.
It writes `casia-codes-2d/` and `casia-codes-christina/` with the convention stem `1` =
gallery (image index 00), stems `2`–`10` = probes (indices 01–09).

### 2. Feature evaluation

```bash
# Stage 1 — DFT bin selection (300-subject synthetic gallery)
python experiments/stage1_spectrum_inspection.py
python experiments/stage1_signal_check.py

# Stage 2 — head-to-head DFT/AC/RL-C9/RL-C6 (2000-subject synthetic gallery)
python experiments/stage2_evaluate.py

# Stage 3 — RRF fusion, consumes Stage 2's saved ranks
python experiments/stage3_fusion.py

# Real-CASIA evaluation at full scale
python experiments/eval_myfeatures_realcasia_full2000.py
python experiments/eval_fusion_realcasia_full2000.py
```

These take no command-line arguments. **`stage3_fusion.py` requires Stage 2 to have run
first** — it re-ranks `results/stage2_ranks.npz`.

### 3. NN training

`features/nn_train.py` trains the `CircularConvEncoder`. It builds the identity split
itself if `--split-path` does not exist (calling `nn_split.build_split` and saving it), so
running `nn_split.py` separately is optional.

The published **v3 checkpoint** (`results/nn_level1_v3_checkpoint_best.pt`, best epoch 18)
was produced by this invocation — recovered from `results/nn_level1_v3_train_history.json`,
whose `hyperparameters` block records `batch_size` 128 with every other value at its
default, and whose `seeds_used` records `split_seed` 0 / `torch_seed` 0:

```bash
python features/nn_train.py --batch-size 128 --out-prefix results/nn_level1_v3 --final-eval
```

That run took ~8.7 hours on CPU (`elapsed_seconds` 31229.6 in the same file).

Flags accepted by `nn_train.py`, with defaults:

| Flag | Default | Flag | Default |
|---|---|---|---|
| `--casia-dir` | `../casia-extraction/casia-codes-2d` | `--margin` | `1.0` |
| `--split-path` | `results/nn_identity_split.json` | `--lr` | `1e-3` |
| `--val-frac` | `0.15` | `--batch-size` | `64` |
| `--test-frac` | `0.15` | `--steps-per-epoch` | `200` |
| `--seed` | `0` | `--epochs` | `20` |
| `--hidden-channels` | `16,32,64` | `--eval-every` | `1` |
| `--embedding-dim` | `64` | `--out-prefix` | `results/nn_level1` |
| `--quant-range` | `127` | `--final-eval` | off (flag) |

`--final-eval` evaluates the best checkpoint on the held-out test split against the full
train+val+test gallery pool. `--hidden-channels` is not recorded in the training history,
but the audit scripts reconstruct the v3 encoder with `hidden_channels=(16, 32, 64)` — the
default — so v3 used the default width.

Then, optionally, the float-vs-quantized comparison:

```bash
python features/nn_eval_quantized.py \
    --checkpoint results/nn_level1_v3_checkpoint_best.pt \
    --out results/nn_level1_v3_quantized_eval.json
```

### 4. Audit scripts

All of these take no arguments unless shown. Most refuse to overwrite an existing output
file (they raise `FileExistsError`), so a re-run needs the previous output moved aside
first. 

```bash
python experiments/audit_nn_v3_verification.py
python experiments/audit_shuffle_test.py
python experiments/audit_all_features_dedup_testsplit.py
python experiments/audit_heldout_gallery_main.py
python experiments/audit_maskleak_main.py
python experiments/audit_maskleak2_main.py
python experiments/audit_maskleak3_main.py
python experiments/audit_embcompare_main.py
python experiments/audit_distributions_main.py
python experiments/audit_ci_recall_v2_vs_v3.py
python experiments/audit_singlescale_regions.py [--quick]
python experiments/audit_singlescale_runlength_hist.py
python experiments/add_hd_histograms.py [--dry-run] [--force]
```

**Ordering constraints.** These are read off the scripts' imports and their file reads, and
are the only ones verifiable from the code:

| Script | Requires first |
|---|---|
| `audit_maskleak2_main.py` | imports `audit_maskleak_main`; reads `results/audit_maskleak_results.json` |
| `audit_maskleak3_main.py` | imports `audit_maskleak_main` and `audit_maskleak2_main`; reads `results/audit_maskleak2_results.json` and `results/audit_all_features_dedup_testsplit.json` |
| `audit_embcompare_main.py` | imports `audit_maskleak_main`; reads `results/audit_all_features_dedup_testsplit.json` |
| `audit_distributions_main.py` | imports `audit_maskleak_main` and `audit_all_features_dedup_testsplit`; reads `results/audit_all_features_dedup_testsplit.json` |
| `audit_ci_recall_v2_vs_v3.py` | needs both `nn_level1_v2_checkpoint_best.pt` and `nn_level1_v3_checkpoint_best.pt` |
| `audit_singlescale_regions.py` | reads `results/hd_distribution_real_vs_synthetic.json` (run `experiments/hd_distribution_real_vs_synthetic.py` first) |
| `audit_singlescale_runlength_hist.py` | reads `results/hd_distribution_real_vs_synthetic.json` and `results/audit_singlescale_results.json` |
| `add_hd_histograms.py` | reads and then edits `results/hd_distribution_real_vs_synthetic.json` in place; also reads `results/audit_singlescale_results.json` |

`audit_maskleak_main.py`, `audit_distributions_main.py`, `audit_embcompare_main.py`,
`audit_ci_recall_v2_vs_v3.py` and `audit_all_features_dedup_testsplit.py` all read
`results/audit_nn_v3_dedup_metrics.json` as their reference baseline and abort if their
recomputation disagrees with it.


`add_hd_histograms.py` is the one script that modifies an existing result file rather than
writing a new one; it is additive, verifies all six stored distributions before writing, and
`--dry-run` reports what it would change without writing.

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
| `hd_distribution_real_vs_synthetic.py` | Bit-level real-vs-synthetic comparison under one method: genuine/impostor fractional-HD distributions (aligned ±*r* and unaligned), degrees of freedom via SIC-Gen's own `mu(1-mu)/sigma^2` estimator, mask valid-bit fraction, and mask-aware run-length distributions, for the full 2000-subject SIC-Gen gallery and all 1998 usable real CASIA identities. All pairs used exhaustively (no sampling) via a matrix-product formulation of fractional HD. Rotation is a full-width roll for SIC-Gen (512 angular positions) and a per-channel roll for real codes (two 256-column phase channels over the same angles); both conventions are measured on the real data and recorded. Supersedes `validate_gen.py`, which covered the synthetic side only, on a 100-subject directory, and saved no numeric output. |
| `audit_singlescale_regions.py` | Tests whether the Gabor scale count and phase-channel layout account for the bit-level real-vs-synthetic gap, by re-measuring unaligned impostor HD, degrees of freedom, mask valid fraction and mask-aware run lengths — estimators imported unmodified from `hd_distribution_real_vs_synthetic.py` — on five slices of the real (32,512) code: full, and each Gabor scale (rows 0–15 coarse, rows 16–31 fine) at both 16×512 (both phase channels) and 16×256 (real channel only). Analysis-only on the existing CASIA export; verifies the full region reproduces the stored real statistics before reporting the slices. |
| `add_hd_histograms.py` | Adds binned HD distributions to `results/hd_distribution_real_vs_synthetic.json` in place (additive only): `bin_edges` / `histogram_density` / `fraction_above_range` on each of the six genuine/impostor distributions, 130 shared bins over HD 0–0.65, density-normalised so the genuine and impostor sets are comparable on one axis. Recomputes the values with the original code and aborts unless all six match the stored n/mean/std/min/max/median, so a plot can be drawn from the measured shape instead of a normal density fitted to mean/std. |
| `audit_singlescale_runlength_hist.py` | Plot-ready run-length distributions for the coarse single-scale single-phase CASIA region (rows 0–15, cols 0–255 = real phase channel — the slice closest to SIC-Gen) against SIC-Gen's whole code, plus the same scale's imaginary channel (cols 256–511) so the phase-channel choice can be checked. `run_length_stats` imported unmodified; also reports degrees of freedom per CASIA channel. Verifies SIC-Gen's mean run length against the stored comparison before writing. |
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
| `audit_maskleak_main.py` | Mask-geometry leakage audit for NN v3: mask-only 32-dim baseline (no model), mask-channel-swap-at-inference ablation (3 seeds, isolating the conv stack's soft-masking pathway from the hard pooling gate), and embedding-distance-vs-mask-distance correlation on impostor pairs — all on the same 299-identity/1998-gallery/2488-dedup-probe protocol as `audit_nn_v3_dedup_metrics.json`. |
| `audit_maskleak2_main.py` | Part 2 of the mask-geometry leakage audit (imports `audit_maskleak_main.py` unmodified): constant-mask (all-ones) channel-1 ablation, mask-dissimilarity stratification of genuine pairs (quartiles of 1-IoU and of row-validity-profile distance, with per-quartile recall/rank/genuine-and-impostor-distance), and partial Pearson/Spearman between embedding and mask-profile distance on impostor pairs controlling for total mask validity. |
| `audit_maskleak3_main.py` | Part 3 (imports `audit_maskleak_main.py`/`audit_maskleak2_main.py` unmodified): repeats Check 5's genuine-pair IoU-dissimilarity quartile stratification for mask-aware DFT/AC/RL-C9, verifying the reconstructed quartile assignment reproduces `audit_maskleak2_results.json`'s stored NN v3 per-quartile numbers exactly before reporting; combined NN-v3-vs-hand-feature table plus each feature's Q1-to-Q4 relative recall drop. |
| `audit_embcompare_main.py` | Representation-overlap and error-correlation audit between NN v3 (quantized, imports `audit_maskleak_main.py` unmodified) and each hand-designed feature (DFT/AC/RL-C9/RL-C6): hand-implemented CCA and OLS regression (no sklearn available), with self-checks (self-pairing ≈1.0, random-pairing low) and condition-number/near-singularity reporting for every whitened/inverted covariance, over the union of gallery+probe vectors; plus per-probe-rank Spearman correlation, Jaccard failure-set overlap at K=10/50, and rescue counts on the matched 299/1998/2488-dedup protocol. |
| `audit_distributions_main.py` | Full genuine/impostor distance distributions for the learned NN v3 embedding (SSD on the quantized embedding) and Christina's enhanced-RL feature (L1, inverted/fixed mask convention) on the matched 299-identity/1998-gallery/2488-dedup-probe protocol (imports `audit_maskleak_main.py`/`audit_all_features_dedup_testsplit.py` unmodified); emits fine-resolution histograms (512 shared-edge bins per feature for overlay, plus 256 own-range bins per set) rather than raw distance arrays, with per-feature bin ranges since the two metrics are not on a common axis, and aborts unless ranks re-derived from the same distance matrices reproduce the published de-duplicated numbers. |
| `render_casia_iris_code_image.py` | Renders a CASIA iris code as a bare binary raster PNG in the `casia_iris_code_example_tall.png` format (unmasked `casia-codes-2d` (32,512) template, bit 1 = white, nearest-neighbour 16x/3x to 1536x512 RGBA, no axes or labels); accepts a CASIA image name (`--image S5079R07.jpg`, mapped to identity/stem via `extract_casia.stem_for_index`) or an explicit `--identity`/`--stem`, and `--verify-format` proves it reproduces the reference PNG pixel-exactly before writing. |

## Results (`results/`)

Saved outputs backing every table and figure. Rank arrays (`stage2_ranks.npz`,
`stage3_fusion_ranks.npz`) are the primary artifacts — every metric and plot can be
recomputed from them without regenerating any iris codes. Also includes the metric
JSONs, the negative-control and row-redundancy data, and the plots.

## Reproducing

See **Setup** for the environment, the three required sibling directories and the CASIA
licensing position, and **Running** for the command for each stage. In short:

- Galleries: regenerable from `../sic-gen` with the recorded seed and process count
  (see each gallery's `metadata.json`), plus the edits in `sicgen_modifications.diff`.
- Environment: `pip install -r requirements.txt`.
- Run scripts from the repository root (paths are anchored via `Path(__file__)`, so
  they resolve regardless of working directory).
- The synthetic results reproduce from this repo and `../sic-gen` alone. The real-data
  results additionally require CASIA-Iris-Thousand and the extraction step
  in `../casia-extraction/`, neither of which is contained in this repo.
