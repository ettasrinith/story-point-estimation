# PS-3 — comparative evaluation

_Generated 2026-11-10 09:40 UTC by `redo/build_comparison.py` from committed results and downloaded Kaggle outputs. No model is trained by this script._

## 1. What is being compared

The assignment (PS-3) asks for **BERT word embeddings, cross-project, random split**: train on one assigned source project, test on a different target project, over the 39 pairs in guideline 3.3. The committed run used a **128-token** BERT window and kept story points in **(0,40]**. The redo repeats the identical protocol with a **256-token** window and the standard **(0,100]** cap.

| | Committed | Redo |
|---|---|---|
| BERT window | 128 tokens | 256 tokens |
| Story-point filter | (0,40] | (0,100] |
| Embedding arrays | `[n,128,768]` float16, 12.2 GB | `[n,256,768]` float16, 24.7 GB |
| Corpus | 62,193 issues / 39 projects | 62,870 issues / 39 projects |
| Split | seed 42, per-project 80:20 | seed 42, per-project 80:20 (recomputed) |
| Model | 3-layer 1-D CNN, masked max pool, Softplus | identical |
| Protocol | 5-fold CV on source pool, best epoch, one held-out target test | identical |

We also keep a **separate extra experiment**: pooled multi-source training (38 sources → 1 target) and its Transformer / MLP predictor ablation, all on the committed corpus. These are clearly labelled `Extension` in every table. They answer a different question and are not PS-3 results.

## 2. Why the corpus and the window changed

### 2.1 Story-point cap

- Raw issues with a story point: **65,427**.
- Kept at (0,40]: **62,193**; kept at (0,100]: **62,870** (+677).
- Dropped for `storypoint > 100`: **141**. The largest observed value is **26,710,114** — an obvious data-entry error, which is exactly why an upper cap is needed.

The guidelines prescribe the split and the 39 pairs but **do not prescribe a story-point filter**, so the cap is a documented design choice. `(0,100]` is the community standard: Deep-SE (Choetkiertikul et al., TSE 2019) drops zero, negative and "unrealistically large" (> 100) values; the TAWOS README sample query uses `Story_Point BETWEEN 1 AND 100`; Tawosi et al. (ESEM 2022) remove SP < 1 and > 100. The original `(0,40]` cap was the outlier.

The cap is not cosmetic. Admitting large story points mechanically raises MAE for every method, including the constant baseline, while MdAE barely moves.

| Corpus | Issues | Assigned-pairs baseline MAE | s.d. | Baseline MdAE | Held-out issues |
|---|---:|---:|---:|---:|---:|
| sp40 | 62,193 | 2.496 | 1.450 | 1.872 | 12,438 |
| sp100 | 62,870 | 2.897 | 2.098 | 1.897 | 12,573 |

### 2.2 Token window

| Window | Issues truncated | Token content lost |
|---|---:|---:|
| 128 tokens | 29.35% | 41.66% |
| 256 tokens | 8.97% | 26.20% |

Measured with the `bert-base-uncased` tokenizer on the (0,100] corpus (`truncation_comparison.json`). Median untruncated length is 81 tokens, p90 243. The longer window clearly preserves more text, but the completed redo does **not** show a macro-MAE gain over its own baseline (§3.2). Because the corpus, cap and partitions all changed, this is not a controlled 128-vs-256 A/B test of the token window.

## 3. Master comparison

### 3.1 Committed run — 128 tokens, (0,40]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.496 | 1.450 | 1.872 | &mdash; | — | 12,438 |
| BERT + CNN (assigned PS-3) | PS-3: 1 source → 1 target | 2.486 | 1.303 | 1.681 | 1.809 | 18/39 | 12,438 |
| Pooled multi-source CNN | Extension: 38 sources → 1 target | 2.191 | 1.340 | 1.414 | &mdash; | 24/39 | 12,438 |
| Pooled Transformer head | Extension: 38 sources → 1 target | 2.242 | 1.260 | 1.511 | &mdash; | 23/39 | 12,438 |
| Pooled MLP head | Extension: 38 sources → 1 target | 2.216 | 1.285 | 1.463 | &mdash; | 23/39 | 12,438 |

### 3.2 Redo — 256 tokens, (0,100]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.897 | 2.098 | 1.897 | &mdash; | — | 12,573 |
| BERT + CNN (redo PS-3) | PS-3: 1 source → 1 target | 2.966 | 2.017 | 1.792 | 1.991 | 15/39 | 12,573 |

**How to read these tables**

- **Test MAE** is the mean of the 39 per-pair test MAEs; **s.d.** is the spread across those pairs. **Test MdAE** is the mean of the 39 per-pair median absolute errors.
- **PS-3: 1 source → 1 target** is the assigned protocol (guideline 3.3). **Extension: 38 sources → 1 target** is a separate extra experiment and must not be read as a PS-3 result or listed as a peer of the PS-3 row.
- **5-fold validation MAE** comes only from the source development pool and is reported separately from the single held-out target test, as required by guideline 3.4.
- The two tables use **different corpora** — 62,193 issues at (0,40] and 62,870 at (0,100]. Compare each model with **its own cohort's baseline**; an old-vs-new MAE difference is descriptive, not a controlled comparison of the token window.


> **What the redo actually shows.** The CNN's macro test MAE (2.966) is **worse** than its median baseline (2.897) by 0.069, and it wins on only 15/39 pairs (pair-level Wilcoxon p = 0.104, not significant). It does improve **typical-case** error: mean per-pair MdAE 1.792 vs 1.897, and issue-weighted MAE 3.961 vs 4.059. Preserving more text is therefore a measured gain in input fidelity, **not** a demonstrated gain in macro MAE.

## 4. Validation and held-out test

Following guideline 3.4, five-fold validation is computed **only** on the source project's development pool and is used solely to choose the best epoch. The target project's held-out 20% is read exactly once, to score the final model. No test row is used for training, early stopping, epoch selection or calibration in the assigned protocol. Validation and test numbers are reported in separate columns above.

A second reason the two cohorts are not paired: `rng.shuffle` walks projects in sorted order from one shared `np.random.default_rng(42)` stream, and the (0,100] corpus changes the row count of 8 projects. Adding rows shifts later draws, so held-out rows move even when the held-out count does not. Within a cohort all models share partitions; across cohorts they do not.

## 5. Provenance and reproduction

| Artefact | Produced by |
|---|---|
| Committed 128 / (0,40] results | `05_run_all_pairs.ipynb`, `06_pooled_training.ipynb`, `07_architecture_ablation.ipynb` |
| 256 / (0,100] embeddings | Kaggle `madmaxop123/tawos-bert-256-a` and `-b` |
| Corpus, baseline, truncation audit | Kaggle `madmaxop123/ps3-256-audit` (CPU) |
| Assigned PS-3 redo | Kaggle `madmaxop123/ps3-256-main` (GPU) |
| Notebook sources | `redo/src/*.py`, assembled by `redo/build_kernels.py` |
| This package | `redo/build_comparison.py` (local aggregation only) |

Reproduce: `python redo/build_kernels.py` → `kaggle kernels push -p redo/kernels/<slug>` → `kaggle kernels output madmaxop123/<slug> -p results/256_sp100/<slug>` → `python redo/build_comparison.py`. All training runs on Kaggle, never locally.

**No results are estimated.** All seven configurations finished on Kaggle; every cell above is a measured value from a completed run (guideline 3.7).
