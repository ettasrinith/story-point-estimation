# Story Point Estimation using BERT Word Embeddings

Natural Language Processing course project — Group 12

## Group members

| Name | Roll No |
|---|---|
| Challa Gouri Shankar | B230648CS |
| Konki Navadeep | B230386CS |
| E Srinith Reddy | B230929CS |
| Fahad Ahmed Mahdi | B230103CS |
| Anita | B230818CS |

## Problem statement

PS-3: Evaluate BERT-based word embeddings for story point estimation when training on source projects and testing on a different target project under the assigned random-split protocol.

In agile teams every issue gets a story point that roughly measures how much work it needs. We build a model that reads the issue text (title + description) and predicts this number. For cross-project estimation the model is trained on one project and tested on a different project it has never seen, and this is repeated for 39 training/testing pairs taken from the project list. Performance is measured with MAE (Mean Absolute Error) and MdAE (Median Absolute Error).

## Dataset

TAWOS (Tawosi et al., MSR 2022, DOI: 10.5522/04/21308124) — Jira issues from 39 open source projects with their story points.

We extracted the data ourselves from the official MySQL dump using `01_extract_tawos.ipynb`. The notebook installs a MySQL server inside the Kaggle notebook, loads the 4.3 GB dump, joins the Issue and Project tables and exports one CSV per project. After dropping issues without story points and cleaning the text we get 65,427 issues. For the experiments we keep story points in the (0, 40] range, which removes 2,416 zero-point tickets and 818 broken values (the largest is 26,710,114) and leaves 62,193 issues. Per-project counts, the cost of that filter and the number of issues whose story point was edited after the estimation date are in `results/manifest.csv` and `results/filter_impact.csv`.

## Repository layout

| Path | What it is |
|---|---|
| `NN_*.ipynb` | The notebooks, as they run on Kaggle. These are what the report quotes. |
| `src/NN_*.py` | The same notebooks as plain Python with `# %%` cell markers - read and edit these, they are the source of truth. |
| `tools/build_notebooks.py` | Converts `src/*.py` into the `.ipynb` files at the repo root. |
| `tools/test_model_cpu.py` | CPU checks that exec the model cells straight out of the notebooks (no GPU, no embeddings). |
| `results/` | Committed outputs of the runs: per-pair CSVs, summaries, plots, data-quality tables. |

```bash
python tools/build_notebooks.py      # src/*.py -> *.ipynb
python tools/test_model_cpu.py       # checks the CNN masking, the head and the fold bookkeeping
```

The notebooks carry no cell outputs: the code is meant to be read as code, and every number it produced is committed under `results/`.

## Progress

- [x] Data extraction from the official dump (39 project CSVs)
- [x] Median baseline over all cross-project pairs
- [x] BERT token embedding extraction
- [x] CNN prediction model with 5-fold validation
- [x] Full 39-pair evaluation and report (see REPORT.md)
- [x] Pooled multi-source model, corrected protocol + target-blind calibration (`06`)
- [x] Architecture ablation: Transformer / MLP vs CNN over the same embeddings (`07`)

## Baseline results

Before training any model we predict the source project's median story point for every issue in the target test set. Any real model has to beat these numbers (seed 42, random 80:20 split):

| Setting | MAE | MdAE |
|---|---|---|
| Cross-project (1482 pairs) | 2.527 ± 1.435 | 1.921 |
| Within-project (39 projects) | 1.926 ± 1.283 | 1.397 |

Full per-pair numbers: `results/baseline_results.csv`. Plot: `results/baseline_by_source.png`.

Cross-project is already ~31% worse than within-project with a single constant prediction, which shows the projects use different story point scales. Shrinking this gap with text representations is the point of the project.

## Notebooks

| Notebook | What it does |
|---|---|
| `01_extract_tawos.ipynb` | Loads the official TAWOS MySQL dump in Kaggle, exports per-project CSVs + manifest |
| `02_baseline_median.ipynb` | Median predictor baseline, MAE/MdAE for every source → target pair |
| `03_bert_embeddings.ipynb` | Frozen bert-base-uncased over every issue, saves token embeddings per project (fp16, [n, 128, 768]) |
| `04_check_embeddings.ipynb` | Verification: all 39 embedding files exist, shapes match metadata, values finite |
| `04_cnn_model.ipynb` | 3-layer CNN regressor over token embeddings; demo pair (Nexus -> CLI) end to end |
| `05_run_all_pairs.ipynb` | **The PS-3 experiment**: 5-fold CV + held-out test on all 39 assigned pairs, compared with the baseline |
| `06_pooled_training.ipynb` | Separate extension, not part of PS-3: pooled multi-source training (GPT2SP-style), log1p target, 3-seed ensemble, target-blind out-of-fold calibration |
| `07_architecture_ablation.ipynb` | Extension: predictor ablation over the same frozen embeddings — small Transformer encoder and mean-pool MLP vs the CNN of 05/06, same pooled protocol |

## Final results (all 39 assigned pairs)

| Model | Test MAE (mean +/- std) | Test MdAE | Beats baseline |
|---|---|---|---|
| Median baseline | 2.496 +/- 1.450 | 1.872 | - |
| Per-source BERT + CNN (`05`) | 2.486 +/- 1.303 | 1.681 | 18 / 39 |
| **Pooled multi-source BERT + CNN (`06`, corrected)** | **2.191 +/- 1.340** | **1.414** | **24 / 39** |

The corrected pooled run (each target trains on the other 38 source projects' development pools,
GPT2SP-style, log1p target, 3-seed ensemble, target-blind grouped-CV epoch selection, no target data
in training/selection/calibration) cuts macro test MAE by 12.2% and the mean per-pair MdAE by 24.5%
vs the baseline, and also beats the like-for-like **pooled median** constant (2.242) on 22/39 pairs.
Issue-level Wilcoxon: p = 2.0e-207 vs the single-source median, p = 6.1e-34 vs the pooled median. Issue-level predictions are in
`results/predictions_pooled/`. The per-source row is the strict PS-3 protocol (one assigned source
per pair). A predictor ablation over the same embeddings (`07`) shows the three-layer CNN is as good
as a small Transformer encoder or a mean-pool MLP (macro MAE 2.195 / 2.203 / 2.221, differences not
significant) — the signal lives in the BERT embeddings and the multi-source pooling, not in the
predictor architecture. Seed 42, story points in (0, 40], same 80:20 partitions for every method. Per-pair
numbers: `results/results_pooled_pairs.csv` (plot: `results/results_pooled_per_pair.png`).
Two names in the assignment table were mapped with the instructor's approval:
Evergreen -> The MongoDB Engineering, C++ driver (Appcelerator) -> Aptana Studio.
Full write-up: [REPORT.md](REPORT.md).

## Embeddings

`03_bert_embeddings.ipynb` (Kaggle GPU) runs bert-base-uncased once over all 62,193 issues and stores
token-level embeddings per project: `emb/<PROJECT>.npy` (float16) with matching `meta/<PROJECT>.csv`
(issue_id, issue_key, storypoint, n_tokens). 12.2 GB total. CNN training reads these files with memory
mapping, so BERT never has to run again.

The 128-token window is the honest weak point of the method. Measured with the `bert-base-uncased`
tokenizer over the published issue texts: **29.5% of issues are longer than 128 tokens** and are cut
off, and those cuts throw away **41.8% of all token content**; the median issue is 81 tokens but the
90th percentile is 244. At 256 tokens the truncated share would drop to 9.0%. It is very uneven
across projects - Atlassian Jira Cloud 92.9% and Jira Server 91.5% of issues truncated, against a
handful of projects near 5%. Numbers: `results/truncation_stats.json` and
`results/truncation_by_project.csv` (measured with the tokenizer over `all_issues_sp.csv`,
2026-11-02; `03` now recomputes them inside the notebook, where earlier revisions had reported this
fraction as ~13-15%).

## How to run

Everything runs on Kaggle; the kernels are `ettasrinithreddy/tawos-*` and each notebook reads the
previous one's output.

| Step | Notebook | Attach as input | Hardware | Kaggle run time |
|---|---|---|---|---|
| 1 | `01_extract_tawos.ipynb` | dataset with `TAWOS.sql` | CPU, internet ON (apt) | not recorded |
| 2 | `02_baseline_median.ipynb` | output of 01 | CPU | not recorded |
| 3 | `03_bert_embeddings.ipynb` | output of 01 | T4 GPU, internet ON (weights) | 2.2 min + verification |
| 4 | `04_check_embeddings.ipynb` | output of 03 | CPU | 0.3 min |
| 5 | `04_cnn_model.ipynb` | outputs of 01 and 03 | T4 GPU | 2.5 min |
| 6 | `05_run_all_pairs.ipynb` | outputs of 01 and 03 | T4 GPU | ~65 min |
| 7 | `06_pooled_training.ipynb` | outputs of 01 and 03 | T4 GPU, 12.2 GB pooled in RAM | ~2-2.5 h (v2) |
| 8 | `07_architecture_ablation.ipynb` | outputs of 01 and 03 | T4 GPU, 12.2 GB pooled in RAM | ~3-4 h |

Times are the ones the kernels on this Kaggle account logged. 01 and 02 have no kernel left on the
account - their outputs are published as the `tawos-extracted` dataset - so run them to get your own.

Steps 5-7 also need `baseline_results.csv` from step 2 (it is part of the 01+02 output dataset) so
that every model is compared against the baseline on the same partitions; each of them asserts that
its own train/test partition matches the baseline before training.

From the command line, once the notebooks are built (`python tools/build_notebooks.py`):

```bash
kaggle kernels push -p <directory with the .ipynb and kernel-metadata.json>
kaggle kernels status <user>/<kernel-ref>
kaggle kernels output <user>/<kernel-ref> -p results/
```
