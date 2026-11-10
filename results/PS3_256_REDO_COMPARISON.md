# PS-3 Redo (256-token, SP ∈ (0,100]) — Preparation & Comparison

This document tracks the **redo configuration** requested after the main PS-3 evaluation: larger label cap and longer token window, evaluated under the assigned per-pair protocol with a predictor ablation.

> **Status:** extraction/ablation **sources, template, split, and configs are prepared and verified locally**. Kaggle GPU runs produce the authoritative result artifacts (`results_ps3_256.csv`, `ps3_256_summary.txt`, `ps3_256_per_pair.png`, `predictions_ps3_256/`, `extended_metrics_ps3_256.csv`); they are not fabricated here and should be added after kernel completion.

## What changes vs. the main runs

| Aspect | Main (05/07) | Redo (11a/11b/12) |
|---|---|---|
| Label filter | SP ∈ (0, 40] | **SP ∈ (0, 100]** (Deep-SE cap; drop SP ≤ 0 and SP > 100) |
| Corpus size | 62,193 issues | **62,870 issues** (39 projects) |
| Token window | 128 | **256** |
| Truncation (corpus-level, measured) | ~29.5% issues, ~41.8% token content lost | **~9% issues truncated** at 256 (exact per-half/per-project stats emitted by 11a/11b) |
| Representation | frozen `bert-base-uncased`, last hidden state, fp16 `[n,128,768]` | frozen `bert-base-uncased`, last hidden state, fp16 **`[n,256,768]`** |
| Protocol | PS-3: 39 assigned pairs, 5-fold CV epoch selection, seed 42, 80:20 | **Same assigned PS-3 protocol** (per-pair, 5-fold CV on source dev pool, best-epoch selection, final train on full pool, eval on target held-out 20%, seed 42) |
| Predictors | CNN (05); CNN/Transformer/MLP pooled ablation (07) | **CNN vs Transformer vs MLP** under the same per-pair protocol (12) |
| Baseline | committed (0,40] partitions/baseline | **Recomputed in-kernel** on the (0,100] partitions (no committed 0,100 baseline) |

This is a **separate configuration** (labels and window both change) — compare via the in-kernel (0,100] baseline and the same protocol/partitions within the run.

## Balanced extraction split (verified)

Computed by `tools/split_256.py` on `all_issues_sp.csv` with filter `(0,100]` (greedy balance by project issue count, largest→smallest round-robin). Manifest: `results/split_256_manifest.json`.

| Half | Kernel/source | Projects | Issues | Est. fp16 output `[n,256,768]` |
|---|---|---:|---:|---:|
| a | `src/11a_extract_bert256_a.py` (`11a_extract_bert256_a.ipynb`) | 17 | 31,333 | ~12.3 GB |
| b | `src/11b_extract_bert256_b.py` (`11b_extract_bert256_b.ipynb`) | 22 | 31,537 | ~12.4 GB |
| **Total** | — | **39** | **62,870** | **~24.7 GB** |

- **Balance:** 204-issue difference between halves; each half kept under the ~20 GB Kaggle kernel-output limit.
- **Verification:** total 62,870; split sums to 24.7 GB; per-half project lists/counts recorded in `results/split_256_manifest.json`.

## Artifacts & run map

| Step | Source / notebook | Purpose | Key outputs |
|---|---|---|---|
| Template | `src/11_extract_bert256.py` | 256-token extraction template (`__HALF__`, `__PROJECTS_JSON__`) | — |
| Split/gen | `tools/split_256.py` | Compute balanced split, regenerate 11a/11b, write manifest | `results/split_256_manifest.json`, `src/11a/11b_*.py` |
| 11a | `src/11a_extract_bert256_a.py` / `.ipynb` | Half-a extraction (GPU) | `emb256/`, `meta/`, `extraction_summary.csv`, `run_metadata.json` |
| 11b | `src/11b_extract_bert256_b.py` / `.ipynb` | Half-b extraction (GPU) | `emb256/`, `meta/`, `extraction_summary.csv`, `run_metadata.json` |
| 12 | `src/12_ps3_redo.py` / `src/12_ps3_256_ablation.py` / `.ipynb` | PS-3 redo ablation (39 pairs × 5-fold × 3 heads) | `results_baseline_100.csv`, `results_ps3_256.csv`, `predictions_ps3_256/`, `extended_metrics_ps3_256.csv`, `ps3_256_summary.txt`, `ps3_256_per_pair.png` |

Configs: `configs/11_extract_bert256.json`, `configs/12_ps3_256_ablation.json`.

## How to run (Kaggle) and collect results

```bash
# rebuild notebooks if sources change
python tools/build_notebooks.py

# (re)compute split + regenerate extraction kernels when data/template changes
python tools/split_256.py <path-to-all_issues_sp.csv>

# push/run kernels (attach inputs per kernel metadata), then download outputs
kaggle kernels push -p kaggle/11a-bert-256-a
kaggle kernels push -p kaggle/11b-bert-256-b
kaggle kernels push -p kaggle/12-ps3-256
kaggle kernels status <user>/<kernel-ref>
kaggle kernels output <user>/<kernel-ref> -p results/ps3_256/
```

After 12 completes, copy the summary/CSVs/plots into `results/` (and predictions into `results/predictions_ps3_256/`) and update the comparison table below with the measured numbers.

## Comparison (fill with kernel results)

| Model / config | Protocol | SP filter | Tokens | Test MAE (macro ± std) | Mean/pool MdAE | Beats baseline | Notes |
|---|---|---|---:|---:|---:|---:|---|
| Median baseline | per-source, 1482/39 | (0,40] | — | 2.496 ± 1.450 | 1.872 | — | `results/baseline_results.csv` |
| BERT + CNN (`05`) | per-source PS-3, 39 pairs | (0,40] | 128 | 2.486 ± 1.303 | 1.681 | 18/39 | `results/results_pairs.csv` |
| TF-IDF+SVD ridge (`08`) | per-source PS-3, 39 pairs | (0,40] | — | 2.235 ± 1.394 | 1.413 | 26/39 | `results/results_tfidf_pairs.csv` |
| Pooled multi-source BERT + CNN (`06`, corrected) | pooled (leave-target-out), 39 pairs | (0,40] | 128 | 2.191 ± 1.340 | 1.414 | 24/39 | `results/results_pooled_pairs.csv` |
| Median baseline (redo) | recomputed in-kernel, 39 pairs | **(0,100]** | — | *(from `results_baseline_100.csv`)* | — | — | 12 output |
| CNN (redo) | PS-3, 39 pairs, 5-fold | **(0,100]** | **256** | *(from `results_ps3_256.csv` / `ps3_256_summary.txt`)* | — | — | 12 output |
| Transformer (redo) | PS-3, 39 pairs, 5-fold | **(0,100]** | **256** | *(from 12)* | — | — | 12 output |
| MLP (redo) | PS-3, 39 pairs, 5-fold | **(0,100]** | **256** | *(from 12)* | — | — | 12 output |

*Placeholders above are intentional until the 12 kernel outputs are available; do not treat as results.*
