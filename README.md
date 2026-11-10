# Story Point Estimation with BERT Word Embeddings

NLP course project — **PS-3**, Group 12

GitHub: https://github.com/ettasrinith/story-point-estimation

## 1. What this project is about

In agile software teams, each issue is given a **story point**: a number that estimates how much
work it needs. We train a model to read an issue's title and description and predict that number.

Our assigned problem statement, **PS-3**, asks one specific question:

> Evaluate **BERT-based word embeddings** for story point estimation when **training on a source
> project** and **testing on a different target project**, using the **assigned random-split
> protocol**.

So for each of the 39 pairs listed in section 3.3 of the guidelines, we train on one project and
test on another project the model has never seen. We measure **MAE** (mean absolute error) and
**MdAE** (median absolute error).

PS-3 covers **BERT word embeddings only**. SBERT sentence embeddings and combined BERT+SBERT
representations are separate problem statements (PS 1–2, 5–12) and are **not** part of this
repository or this report.

## 2. Group members

| Name | Roll No |
|---|---|
| Challa Gouri Shankar | B230648CS |
| Konki Navadeep | B230386CS |
| E Srinith Reddy | B230929CS |
| Fahad Ahmed Mahdi | B230103CS |
| Anita | B230818CS |

## 3. Dataset

**TAWOS** (Tawosi et al., MSR 2022; DOI `10.5522/04/21308124`) — Jira issues with story points
from 39 open-source projects. We loaded the official MySQL dump ourselves in
`01_extract_tawos.ipynb`, joined the Issue and Project tables, stripped HTML from the text, and
exported one CSV per project. That gives **65,427 issues with a story point**.

Story points contain zeros and obvious data-entry errors (the largest is **26,710,114**), so a
filter is necessary. The guidelines do **not** prescribe one, so we treat it as a documented
design choice and report two corpora:

| | Committed run | Redo |
|---|---|---|
| Filter | `(0,40]` | `(0,100]` |
| Issues kept | 62,193 | 62,870 (+677) |
| Dropped above the cap | 818 | 141 |
| BERT window | 128 tokens | 256 tokens |

`(0,100]` is the standard upper bound used in the literature — Deep-SE (Choetkiertikul et al., TSE 2019),
the TAWOS README sample query, and Tawosi et al. (ESEM 2022) all discard story points above 100.
The original `(0,40]` cap was the outlier. See `results/256_sp100/audit/audit_corpus.json` for the
exact counts and `results/filter_impact.csv` for the per-project cost of the filter.

## 4. Two different experiments — please keep them separate

This repository contains **one assigned experiment** and **one extra experiment**. They answer
different questions and their numbers are never mixed.

| | Assigned PS-3 | Pooled extension |
|---|---|---|
| Notebooks | `05_run_all_pairs` | `06_pooled_training`, `07_architecture_ablation` |
| Training data per target | **one** assigned source project | the **other 38** projects combined |
| Question | does BERT transfer from one project to another? | does more source data help? |
| Status | the required deliverable | extra analysis, clearly labelled |

The extension also changes the target to `log1p(storypoint)` and averages three seeds, so its
improvement cannot be attributed to pooling alone. Every results table below has a **Protocol**
column that makes this distinction explicit.

## 5. Repository layout

| Path | Contents |
|---|---|
| `configs/ps3.yaml` | **All** model names, hyperparameters, seeds and split settings (guideline 3.5) |
| `src/*.py` | Readable Python sources of the notebooks, with `# %%` cell markers — edit these |
| `NN_*.ipynb` | The generated notebooks, as they run on Kaggle |
| `tools/build_notebooks.py` | Converts `src/*.py` → `.ipynb` |
| `tools/test_model_cpu.py` | CPU checks for the CNN padding mask, head and fold bookkeeping |
| `tools/verify_project_mapping.py` | Machine-checks the 39 PDF pair names against TAWOS |
| `redo/` | The 256-token / (0,100] redo: kernel builders and the reporting script |
| `results/` | Per-pair metrics, summaries, plots, predictions, audits |
| `results/comparison/` | The neat comparative package (tables, dashboard, CSVs) |
| `REPORT.md` / `REPORT.pdf` | Final report |

Notebooks carry no saved cell outputs; every number they produced is committed under `results/`.

## 6. Results

<!-- BEGIN GENERATED RESULTS -->
_Auto-generated 2026-11-10 09:40 UTC by `redo/build_comparison.py`. Do not edit between the markers._

### Committed run — 128 tokens, story points (0,40]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.496 | 1.450 | 1.872 | &mdash; | — | 12,438 |
| BERT + CNN (assigned PS-3) | PS-3: 1 source → 1 target | 2.486 | 1.303 | 1.681 | 1.809 | 18/39 | 12,438 |
| Pooled multi-source CNN | Extension: 38 sources → 1 target | 2.191 | 1.340 | 1.414 | &mdash; | 24/39 | 12,438 |
| Pooled Transformer head | Extension: 38 sources → 1 target | 2.242 | 1.260 | 1.511 | &mdash; | 23/39 | 12,438 |
| Pooled MLP head | Extension: 38 sources → 1 target | 2.216 | 1.285 | 1.463 | &mdash; | 23/39 | 12,438 |

### Redo — 256 tokens, story points (0,100]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.897 | 2.098 | 1.897 | &mdash; | — | 12,573 |
| BERT + CNN (redo PS-3) | PS-3: 1 source → 1 target | 2.966 | 2.017 | 1.792 | 1.991 | 15/39 | 12,573 |


**How to read these tables**

- **Test MAE** is the mean of the 39 per-pair test MAEs; **s.d.** is the spread across those pairs. **Test MdAE** is the mean of the 39 per-pair median absolute errors.
- **PS-3: 1 source → 1 target** is the assigned protocol (guideline 3.3). **Extension: 38 sources → 1 target** is a separate extra experiment and must not be read as a PS-3 result or listed as a peer of the PS-3 row.
- **5-fold validation MAE** comes only from the source development pool and is reported separately from the single held-out target test, as required by guideline 3.4.
- The two tables use **different corpora** — 62,193 issues at (0,40] and 62,870 at (0,100]. Compare each model with **its own cohort's baseline**; an old-vs-new MAE difference is descriptive, not a controlled comparison of the token window.

Full comparative package: [`results/comparison/`](results/comparison/COMPARISON.md).
<!-- END GENERATED RESULTS -->

## 7. Why the 256-token redo

A 128-token BERT window is too short for a large share of TAWOS issues. Measured with the same
tokenizer on the `(0,100]` corpus:

| Window | Issues truncated | Token content lost |
|---|---:|---:|
| 128 tokens | 29.35% | 41.66% |
| 256 tokens | 8.97% | 26.20% |

So the longer window clearly preserves more text. **The finished CNN run does not turn that into a
macro-MAE win**: it scores 2.966 against its own 2.897 median baseline, winning on only 15 of 39
pairs (pair-level Wilcoxon p = 0.104, not significant). It does improve typical-case error — MdAE
1.792 vs 1.897, and issue-weighted MAE 3.961 vs 4.059. We report this as a **measured gain in
input fidelity, not a demonstrated gain in accuracy**.

Note that the redo changed the story-point cap and the corpus partitions as well as the token
window, so it is **not** a controlled 128-vs-256 A/B comparison. Evidence:
`results/256_sp100/audit/truncation_comparison.json` and
`results/256_sp100/main/master_metrics_256.json`.

## 8. Project-name mapping

Section 3.3 of the guidelines uses short/informal project names. `tools/verify_project_mapping.py`
resolves them in three machine-checked steps: **13 exact** matches, **24 normalised** short forms
(e.g. `MESOS` → `Apache Mesos`), and **2 instructor-approved remaps** for labels that name no TAWOS
project:

- `Evergreen` → `The MongoDB Engineering`
- `C++ driver (Appcelerator)` → `Aptana Studio`

After exact + normalised matching, exactly those two TAWOS projects remain, so the mapping is a
verified **bijection**: 39 PDF labels → 39 distinct TAWOS projects. Outputs:
`results/project_name_mapping.csv` and `results/project_mapping_validation.json`.

## 9. Reproducing the experiments

**All training runs on Kaggle GPUs; nothing is trained locally.** The Kaggle account is
`madmaxop123`; the extracted dataset is `ettasrinithreddy/tawos-extracted`.

```bash
# 1. Build notebooks from the readable sources
python tools/build_notebooks.py
python tools/test_model_cpu.py

# 2. Rebuild the redo kernels and push them to Kaggle
python redo/build_kernels.py
kaggle kernels push -p redo/kernels/ps3-256-audit      # CPU: corpus / baseline / truncation
kaggle kernels push -p redo/kernels/ps3-256-main       # GPU: the 39-pair BERT + CNN redo

# 3. Download finished outputs
kaggle kernels output madmaxop123/ps3-256-audit -p results/256_sp100/audit
kaggle kernels output madmaxop123/ps3-256-main  -p results/256_sp100/main

# 4. Regenerate every results table, the dashboard and the report's results section
python redo/build_comparison.py

# 5. One-shot shortcut: download outputs + rebuild every table and the PDF
python redo/finalize_main.py
```

The committed `05`/`06`/`07` runs reproduce from `03_bert_embeddings.ipynb` (frozen
`bert-base-uncased`, fp16 token arrays) followed by the respective training notebooks. Full
hyperparameters and seeds: `configs/ps3.yaml`. Step-by-step Kaggle inputs, hardware and runtimes
are in `REPORT.md` section 8.
