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

We extracted the data ourselves from the official MySQL dump using `01_extract_tawos.ipynb`. The notebook installs a MySQL server inside the Kaggle notebook, loads the 4.3 GB dump, joins the Issue and Project tables and exports one CSV per project. After dropping issues without story points and cleaning the text we get 65,427 issues. For the experiments we keep story points in the (0, 40] range (removes broken values like 26 million) which gives 62,193 issues. Per-project counts are in `results/manifest.csv`.

## Progress

- [x] Data extraction from the official dump (39 project CSVs)
- [x] Median baseline over all cross-project pairs
- [x] BERT token embedding extraction
- [x] CNN prediction model with 5-fold validation
- [x] Full 39-pair evaluation and report (see REPORT.md)

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
| `05_run_all_pairs.ipynb` | Full evaluation: 5-fold CV + held-out test on all 39 assigned pairs, compared with the baseline |

## Final results (all 39 assigned pairs)

| Metric | BERT + CNN | Median baseline |
|---|---|---|
| Test MAE (mean +/- std over pairs) | **2.326 +/- 1.350** | 2.496 +/- 1.450 |
| Test MdAE (mean over pairs) | **1.710** | 1.872 |
| Mean 5-fold validation MAE | 1.923 +/- 0.213 | - |
| Pairs beating the baseline | 27 / 39 | - |

Seed 42, story points in (0, 40], same 80:20 partitions for both methods, 5-fold CV on the source
train pool, single evaluation on the held-out target test set. Per-pair numbers:
`results/results_pairs.csv`, per-pair plot: `results/results_per_pair.png`.
Two names in the assignment table were mapped with the instructor's approval:
Evergreen -> The MongoDB Engineering, C++ driver (Appcelerator) -> Aptana Studio.
Full write-up: [REPORT.md](REPORT.md).

## Embeddings

`03_bert_embeddings.ipynb` (Kaggle GPU) runs bert-base-uncased once over all 62,193 issues and stores
token-level embeddings per project: `emb/<PROJECT>.npy` (float16) with matching `meta/<PROJECT>.csv`
(issue_id, issue_key, storypoint, n_tokens). 12.2 GB total. Max length is 128 tokens, so about 15% of
the longest descriptions get truncated (fraction is in `extraction_summary.csv`). CNN training reads
these files with memory mapping, BERT never has to run again.

## How to run

Both notebooks run on Kaggle (CPU only so far):

1. Upload `TAWOS.sql` as a private Kaggle dataset.
2. Open `01_extract_tawos.ipynb` on Kaggle, attach the dataset as input, Run All. It produces `all_issues_sp.csv`, `csv/` (one file per project) and `manifest.csv`.
3. Open `02_baseline_median.ipynb`, attach the output of step 2 as input, Run All. It produces `baseline_results.csv` and the summary.

Internet is only needed in the first notebook (to install MySQL) and the third (to download the BERT weights). Only the third needs a GPU (T4, ~8 minutes).
