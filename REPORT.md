# Story Point Estimation with BERT Word Embeddings — Final Report

**PS-3: Evaluate BERT-based word embeddings for story point estimation when training on source
projects and testing on a different target project under the assigned random-split protocol.**

Group 12 — Challa Gouri Shankar (B230648CS), Konki Navadeep (B230386CS), E Srinith Reddy (B230929CS),
Fahad Ahmed Mahdi (B230103CS), Anita (B230818CS)

GitHub: https://github.com/ettasrinith/story-point-estimation

## 1. Problem

Agile teams label each issue with a story point — a rough measure of the work it needs. We study
whether a model that reads the issue text can predict this number, and specifically whether such a
model **transfers across projects**: train on one project's issues, test on a different project it
never saw. This is repeated over the 39 training/testing pairs assigned in the project list, using
BERT word embeddings as the text representation, a three-layer CNN as the predictor, random
80:20 splitting, and MAE / MdAE as the metrics.

## 2. Dataset

TAWOS (Tawosi et al., MSR 2022; DOI 10.5522/04/21308124) — 65427 Jira issues with story
points across 39 open-source projects (extracted from the official MySQL dump by
`01_extract_tawos.ipynb`; per-project counts in `results/manifest.csv`).

Cleaning: story points cast to numeric, issues without a story point dropped (458,232 → 65,427),
HTML markup stripped from title + description, duplicates removed (no duplicate `issue_id` and no
empty text remains in the 65,427). For all experiments story points are kept in (0, 40] — this removes
2,416 zero-point tickets and 818 values above 40, including broken ones (one project contains a story
point of 26,710,114) — leaving 62,193 issues; the per-project cost of that filter is in
`results/filter_impact.csv`. Project sizes range from 209 (Apache MXNet) to 19,358 (Lsstcorp Data
management) issues. TAWOS also records whether an estimate was edited afterwards: 8,159 issues (12.5%)
have a story point that changed after the estimation timestamp. Those rows are kept — the label we are
asked to predict is the final one — but the count is reported per project in `filter_impact.csv`
because it bounds how noisy the target variable is.

## 3. Method

**Representation.** `bert-base-uncased`, frozen, inference only. Each issue's text
(title + description) is tokenized (max length 128) and encoded once; the token-level last-hidden-state
vectors [128 × 768] are stored per project (float16, 12.2 GB total; `03_bert_embeddings.ipynb`).
128 tokens is not generous: measured with the same tokenizer over all 62,193 issues, 29.5% of them are
longer than the window and get cut, which discards 41.8% of the token content of the corpus (median
issue 81 tokens, 90th percentile 244). The effect is concentrated in the Atlassian projects — 92.9% of
Jira Cloud and 91.5% of Jira Server issues are truncated (`results/truncation_stats.json`,
`results/truncation_by_project.csv`).

**Model.** Three-layer 1D CNN (768→256→128→64 channels, kernel 3, ReLU, dropout 0.2 on the first
layer) over the token embeddings, masked global max pooling over real tokens, then a small MLP head
(64→32→1) with a Softplus output, so a story point is never predicted negative. The token mask is
applied **before** the first convolution and re-applied after every layer, so padded positions cannot
reach the pooling through a convolution's left context. L1 loss (matches MAE), Adam lr 1e-3, batch 64,
early stopping on validation MAE (patience 4, max 25 epochs). fp16 autocast on a T4 GPU.

**Protocol (per pair).** Each project's issues are split randomly 80:20 with seed 42 — the same
partitions used by the baseline, and each training notebook *verifies* this before running: it compares
the median of every source project's train pool with the constant the baseline predicted from that
pool, which catches a partition mismatch that equal set sizes would not. (1) 5-fold cross-validation
on the source project's 80% pool: train on 4 folds, validate on 1; each fold keeps the weights of its
best validation epoch and contributes that validation MAE and that epoch number. (2) The final model
trains on the full pool for the mean of the five best epochs. (3) Single evaluation on the target
project's held-out 20%. The target project's data never touches training or model selection.

**Baseline.** Predict the source project's median story point for every target issue
(`02_baseline_median.ipynb`).

**Corrections in this revision.** The first version of notebooks 04-06 reported the *last* validation
epoch of each fold instead of the best one, used the epoch at which early stopping fired as the fold's
"best epoch", let padded tokens into the convolutions, and allowed negative predictions. It also
built its splits wrong: `Index.to_numpy()` returns the index's own buffer on the Kaggle image's pandas,
so shuffling the draw reordered the label array that the per-project ranks were then read from, and the
ranks collapsed onto the shuffle order — the "test set" became **the first 20% of each project's rows
in file order** instead of the seed-42 random 20%. Every committed CNN number therefore came from a
different partition than the baseline it was compared against (the model's test rows were a positional
cut, the baseline's a random one). The median check described above is what caught it. The per-source
experiment (05) and the demo pair (04) have been re-run with the corrected code, and the pooled
experiment (06) with the corrected and extended code. Notebook 06 additionally had a 6-epoch
ceiling that every cross-validation fold reached, so its epoch selection was a no-op; that ceiling is
now 12 and the folds stop on their own.

Two project names in the assignment table have no exact match in the dataset ("Evergreen",
"C++ driver (Appcelerator)"); with the instructor's approval they are mapped to the two remaining
TAWOS projects, "The MongoDB Engineering" and "Aptana Studio".

## 4. Results

**Assigned-protocol result: per-source BERT+CNN (05, corrected).** For each assigned pair the model
trains on one source project's development pool and is evaluated once on the target's held-out 20% —
this is the PS-3 experiment, re-run with the corrected protocol (best-epoch fold selection, padding
mask before every convolution, partitions verified against the baseline).

**Multi-source extension (06) — corrected and extended.** Following GPT2SP (Fu & Tantithamthavorn,
2023), each target model trains on the combined train pools of the other 38 projects, learns
log1p(story point) and averages 3 seeds — the target's own data never enters its model's training,
epoch selection (project-grouped CV folds; folds that held the target out are excluded from that
target's average) or calibration. The revision adds a **target-blind affine calibration** `a·p + b`
per target, fitted on the other 38 projects' out-of-fold grouped-CV predictions only, which corrects
the systematic underestimation that `expm1(log1p(·))` regression introduces on skewed labels; paired
Wilcoxon tests against both constant baselines; and true best-epoch selection under a 10-epoch CV
cap (the first build's 6-epoch cap was a no-op).

| Model | Test MAE (mean ± std over pairs) | Test MdAE (mean over pairs) | Beats baseline |
|---|---|---|---|
| Median baseline | 2.496 ± 1.450 | 1.872 | — |
| Per-source BERT+CNN (05) | 2.486 ± 1.303 | 1.681 | 18 of 39 |
| **Pooled multi-source BERT+CNN (06, corrected)** | **2.191 ± 1.340** | **1.414** | **24 of 39** |

Against the same-protocol baseline, the corrected per-source model improves the mean per-pair MdAE by
10.2% (1.681 vs 1.872) and the issue-weighted MAE by 4.2% (3.185 vs 3.323), while the pair-average
MAE is essentially tied (2.486 vs 2.496): the gain concentrates in the typical issue rather than in
the pairs with the largest scales. It beats the baseline on 18 of the 38 pairs where a baseline is
beatable (the 39th, C++ driver (MongoDB) → Java driver, has baseline MAE 0 — every held-out issue
there carries the same story point). Pooling the other 38 source projects is what turns the tie into
a clear win: macro MAE 2.191 (−12.2% vs the baseline), mean per-pair MdAE 1.414 (−24.5%), and
issue-weighted MAE 2.741 (−17.5%), with the target's development and test data fully excluded from
its own model, epoch selection and calibration. Against the like-for-like constant — the **pooled
median** of the same 38 projects (macro 2.242) — the model wins on 22 of 39 pairs. Issue-level
Wilcoxon tests over the 12,438 held-out issues: p = 2.0e-207 vs the single-source median and
p = 6.1e-34 vs the pooled median; per-pair across the 39 pairs, p = 0.044 vs the single-source
median. The target-blind affine calibration turned out to be a no-op on this task (macro 2.191 →
2.195), i.e. the log1p training was already near-unbiased — reported rather than claimed as a gain.

Per-pair pooled results: `results/results_pooled_pairs.csv` (plot: `results/results_pooled_per_pair.png`).

### 4.1 Per-source model (the assigned PS-3 experiment, all 39 pairs, corrected protocol)

| Metric | BERT + CNN | Median baseline |
|---|---|---|
| Test MAE (mean ± std over pairs) | 2.486 ± 1.303 | 2.496 ± 1.450 |
| Test MdAE (mean over pairs) | 1.681 | 1.872 |
| Issue-weighted test MAE | 3.185 | 3.323 |
| Pooled MdAE over all 12,438 held-out issues | 1.887 | — |
| Mean 5-fold validation MAE (best epoch) | 1.809 ± 0.221 | — |
| Epochs for the final model (mean of fold best epochs) | 8.8 (range 2–16) | — |
| Pairs where CNN beats baseline MAE | 18/39 (1 pair has baseline MAE 0) | — |

Issue-level predictions for every pair are in `results/predictions/`, which is what the pooled MdAE
and the issue-weighted average are computed from.

Per-pair results (validation from 5-fold CV on the source; test = held-out target):

| Source | Target | Val MAE (5-fold) | Test MAE | Test MdAE | Baseline MAE |
|---|---|---|---|---|---|
| XD | MESOS | 1.771 ± 0.119 | 1.624 | 0.992 | 1.538 |
| Usergrid | MXNet | 0.933 ± 0.192 | 2.531 | 1.083 | 2.425 |
| Titanium Mobile Platform | Usergrid | 2.294 ± 0.179 | 2.667 | 1.870 | 0.729 |
| The Titanium SDK | Alloy Framework | 2.773 ± 0.107 | 2.370 | 2.063 | 2.180 |
| Software Cloud | Appcelerator Daemon | 2.022 ± 0.270 | 3.351 | 1.819 | 3.375 |
| Sawtooth | Appcelerator Studio | 0.911 ± 0.047 | 4.044 | 3.529 | 3.646 |
| Nexus | Command-Line Interface | 0.962 ± 0.095 | 2.425 | 1.795 | 2.789 |
| MXNet | The Titanium SDK | 2.635 ± 0.387 | 3.507 | 2.010 | 3.470 |
| Mule APIkit | Titanium Mobile Platform | 1.760 ± 0.302 | 3.340 | 1.737 | 3.168 |
| Mule | C++ driver (Appcelerator) | 2.571 ± 0.097 | 5.439 | 3.634 | 5.880 |
| Moodle | Atlassian Software Server | 6.262 ± 0.684 | 2.586 | 1.633 | 2.492 |
| MongoDB Core Server | Bamboo | 1.320 ± 0.167 | 1.347 | 0.863 | 1.277 |
| MESOS | Clover | 1.362 ± 0.070 | 4.167 | 1.616 | 4.178 |
| Lyrasis Dura Cloud | Confluence Cloud | 0.934 ± 0.137 | 2.211 | 1.045 | 2.390 |
| Lsstcorp Data management | Confluence Server | 3.112 ± 0.074 | 1.810 | 1.561 | 1.039 |
| Jira Server | Crowd | 1.744 ± 0.230 | 1.714 | 1.884 | 2.558 |
| Jira Cloud | FishEye | 1.280 ± 0.287 | 1.566 | 1.586 | 2.169 |
| Java driver | Jira Cloud | 0.128 ± 0.143 | 3.270 | 3.026 | 3.243 |
| Indy SDK | Jira Server | 1.546 ± 0.120 | 1.983 | 1.156 | 1.987 |
| Indy Node | Software Cloud | 1.415 ± 0.165 | 2.374 | 1.173 | 2.237 |
| FishEye | DotNetNuke Platform | 0.867 ± 0.109 | 1.023 | 0.907 | 1.040 |
| Fabric | Blockchain Explorer | 1.408 ± 0.278 | 1.777 | 1.512 | 1.501 |
| Evergreen | Fabric | 0.732 ± 0.027 | 1.712 | 0.728 | 1.752 |
| DotNetNuke Platform | Indy Node | 0.882 ± 0.086 | 1.871 | 1.333 | 1.956 |
| DataCass | Indy SDK | 2.442 ± 0.731 | 1.865 | 0.841 | 1.620 |
| Crowd | Sawtooth | 0.922 ± 0.122 | 1.184 | 0.601 | 1.368 |
| Confluence Server | Lsstcorp Data management | 1.233 ± 0.132 | 3.961 | 2.037 | 3.886 |
| Confluence Cloud | Lyrasis Dura Cloud | 1.268 ± 0.202 | 1.429 | 1.076 | 1.878 |
| Compass | MongoDB Core Server | 1.210 ± 0.046 | 1.420 | 1.392 | 1.490 |
| Command-Line Interface | Compass | 1.736 ± 0.143 | 1.528 | 1.162 | 1.300 |
| Clover | C++ driver (MongoDB) | 2.849 ± 0.690 | 0.913 | 0.838 | 0.917 |
| C++ driver (Appcelerator) | Evergreen | 3.606 ± 0.508 | 3.561 | 3.503 | 6.099 |
| C++ driver (MongoDB) | Java driver | 0.487 ± 0.161 | 0.027 | 0.028 | 0.000 |
| Blockchain Explorer | Moodle | 1.106 ± 0.155 | 7.056 | 2.318 | 6.858 |
| Bamboo | Mule | 1.146 ± 0.192 | 3.101 | 1.805 | 3.150 |
| Atlassian Software Server | Mule APIkit | 2.004 ± 0.188 | 1.961 | 1.479 | 1.882 |
| Appcelerator Studio | Nexus | 2.097 ± 0.070 | 3.273 | 3.275 | 3.390 |
| Appcelerator Daemon | DataCass | 5.135 ± 0.831 | 2.782 | 2.976 | 2.401 |
| Alloy Framework | XD | 1.687 ± 0.085 | 2.171 | 1.666 | 2.079 |
Best pairs: C++ driver (MongoDB)→Java driver (0.027, the pair whose baseline MAE is 0),
Clover→C++ driver (MongoDB) (0.913), FishEye→DotNetNuke Platform (1.023).
Worst pairs: Blockchain Explorer→Moodle (7.056), Mule→C++ driver (Appcelerator) (5.439), MESOS→Clover (4.167).

Demo pair walkthrough (Nexus → Command-Line Interface, `04_cnn_model.ipynb`): the five folds' best
validation MAEs are 0.962 ± 0.095 at epochs 8, 15, 22, 6 and 5, so the final model trains for 11
epochs on the full pool and scores test MAE 2.425 (MdAE 1.795) on the 71 held-out target issues,
against the baseline's 2.789. Per-fold curves, predicted-vs-actual and the error histogram are in
`results/demo_pair_plots.png`.

### 4.2 Predictor architecture ablation (`07_architecture_ablation.ipynb`)

Is the three-layer CNN the right predictor over the frozen BERT tokens, or would a small Transformer
encoder (or a plain mean-pool MLP) do better? Guideline 3.1 allows a three-layer CNN as the
prediction component — *may*, not *must* — so this ablation holds the representation, protocol,
folds, seeds and calibration scheme fixed and varies only the predictor (all three have a comparable
0.2–0.9 M parameter budget and the same Softplus head):

| Predictor (pooled protocol, calibrated) | Macro test MAE | Issue-weighted MAE | Beats source median | Beats pooled median | Wilcoxon vs CNN |
|---|---|---|---|---|---|
| 3-layer CNN (06) | 2.195 | 2.758 | 24/39 | 22/39 | — |
| Transformer encoder (1 layer, d=256, 4 heads) | 2.203 | 2.761 | 24/39 | 23/39 | p = 0.92 |
| MLP over masked mean-pooled tokens | 2.221 | 2.748 | 23/39 | 22/39 | p = 0.31 |

The three predictors are statistically indistinguishable (CNN vs Transformer p = 0.92; median
per-pair spread 0.10 MAE points; Transformer vs MLP p = 0.15). Two conclusions follow. First, the
three-layer CNN is kept: it satisfies the guideline and costs nothing in accuracy. Second, and more
interesting, even a plain mean-pool MLP recovers essentially all of the performance — the predictive
signal lives in the frozen BERT token embeddings themselves, and pooling sources multiplies it, while
the predictor's inductive bias (local convolutions vs global attention vs none) contributes little at
this dataset scale. Per-pair numbers: `results/results_pooled_transformer.csv`,
`results/results_pooled_mlp.csv`; issue-level predictions in `results/predictions_arch/`.

## 5. Discussion

- The corrected per-source CNN improves the baseline's mean per-pair MdAE by 10.2% (1.872 → 1.681)
  and its issue-weighted MAE by 4.2% (3.323 → 3.185), and beats it on 18 of the 38 beatable pairs —
  while the pair-average MAE is a tie (2.486 vs 2.496). BERT word embeddings carry real but modest
  effort-relevant signal under single-source cross-project transfer; the earlier 27/39 figure came
  from the faulty split and was withdrawn.
- Pooling the other 38 source projects converts that tie into the experiment's clearest result:
  macro MAE 2.191 (−12.2% vs baseline), per-pair MdAE 1.414 (−24.5%), issue-weighted MAE 2.741
  (−17.5%), 24/39 over the single-source median and 22/39 over the pooled median constant
  (macro 2.242) that sees the same training data. The pooled model needs only 1–2 epochs
  (grouped-CV best epochs), against ~9 for single-source training — 48k pooled rows converge almost
  immediately. This matches the literature: GPT2SP's cross-project gains also come from multi-source
  training, not from single-source transfer.
- Per-source validation MAE (1.809) is well below its test MAE (2.486): the model is accurate
  *within* the source project but degrades on unseen projects. The transfer gap, not the
  architecture, is the binding constraint of the single-source setting.
- The target-blind affine calibration was a no-op (2.191 → 2.195): the log1p target already made the
  pooled model near-unbiased, so scale mismatch is *not* a global bias the calibration could remove —
  it is per-pair variance (worst pairs remain pairs whose target scale differs wildly from the pool).
- Worst pairs combine a source and target with very different story-point scales (e.g. a source with
  median 1–2 and a target with median 5); text alone cannot recover the target's scale, so the model
  inherits a systematic offset. Best pairs share vocabulary domains and scales.
- Validation results are reported separately from held-out test results, per §3.4; the final
  configuration (epochs) was selected using development folds only, target-blind for every model.

## 6. Threats to validity

- **Construct:** story points are team-specific estimates, not objective effort; the (0, 40] filter
  and per-project scales make cross-project MAE hard to interpret in absolute terms. 12.5% of the
  issues kept had their story point edited after the estimation timestamp, so part of the label noise
  is in the target itself, not in the text.
- **Representation:** the 128-token window truncates 29.5% of issues and 41.8% of the corpus' token
  content, and it does so unevenly (92.9% of Jira Cloud issues lose their tail). A project whose
  descriptions are long is penalised by the representation, not by its difficulty.
- **Internal:** fixed seeds and partitions for all methods; every training notebook checks its
  partitions against the baseline's recorded medians before it trains; model selection used development
  folds only; the target project never influenced training. The two instructor-approved name mappings
  ("Evergreen" → "The MongoDB Engineering", "C++ driver (Appcelerator)" → "Aptana Studio") are the
  only interpretive steps in the pair list.
- **External:** 39 pairs from one dataset (open-source Jira projects); results may not generalize to
  commercial trackers or other estimation scales.
- **Statistical:** one run per pair with a fixed seed; fold variance is reported, but no repeated
  runs with different global seeds. With 39 pairs and a pair-level std of ~1.3, the difference from
  the baseline is not individually significant per pair.

## 7. Conclusion

Under the corrected protocol, BERT word embeddings with a small CNN beat the median baseline on 18 of
the 38 beatable assigned pairs when trained on a single source (MdAE −10.2%, pair-average MAE tied)
and on 24 of 39 when trained on the pooled development data of the other 38 source projects —
macro MAE 2.191 vs 2.496 (−12.2%), per-pair MdAE 1.414 vs 1.872 (−24.5%), issue-weighted MAE 2.741
vs 3.323 (−17.5%), significant at issue level against both constant baselines (Wilcoxon
p ≤ 6.1e-34). With no target data anywhere in training, selection or calibration, the transfer gap
itself (validation 1.809 vs test 2.486 in the single-source setting) remains the dominant obstacle —
combining text with target-project calibration is the natural next step.

## Reproducibility

The notebooks are generated from the plain-Python sources in `src/` (`python tools/build_notebooks.py`);
read those if the `.ipynb` JSON is in the way. `tools/test_model_cpu.py` re-executes the model cells of
04/05/06 on CPU and asserts the properties the fixes were supposed to buy (pads inert, head
non-negative, a fold reporting its best epoch with the weights that scored it).

| Notebook | Kaggle kernel | Hardware | Inputs | Outputs | Run time |
|---|---|---|---|---|---|
| `01_extract_tawos` | (published as dataset `tawos-extracted`) | CPU, internet ON | dataset with `TAWOS.sql` | `all_issues_sp.csv`, `csv/`, `manifest.csv`, `filter_impact.csv`, `eda.png` | — |
| `02_baseline_median` | — | CPU | output of 01 | `baseline_results.csv`, `baseline_summary.txt`, `baseline_by_source.png` | — |
| `03_bert_embeddings` | `ettasrinithreddy/tawos-bert-embeddings` | T4, internet ON | output of 01 | `emb/*.npy`, `meta/*.csv`, `extraction_summary.csv`, `run_metadata.json` | 2.2 min |
| `04_check_embeddings` | `ettasrinithreddy/tawos-check-emb` | CPU | output of 03 | verification table | 0.3 min |
| `04_cnn_model` | `ettasrinithreddy/tawos-cnn-demo-v2` | T4 | outputs of 01 + 03 | `demo_pair_result.csv`, `demo_pair_plots.png` | 1 min |
| `05_run_all_pairs` | `ettasrinithreddy/tawos-all-pairs-v2` | T4 | outputs of 01 + 03 | `results_pairs.csv`, `evaluation_summary.txt`, `results_per_pair.png`, `predictions/*.csv` | see summary |
| `06_pooled_training` | `ettasrinithreddy/tawos-pooled-v2` (corrected + calibrated) | T4, 12.2 GB RAM | outputs of 01 + 03 | `results_pooled_pairs.csv`, `evaluation_summary_pooled.txt`, plot, `predictions_pooled/*.csv` | ~2.5 h |
| `07_architecture_ablation` | `ettasrinithreddy/tawos-arch-ablation` | T4, 12.2 GB RAM | outputs of 01 + 03 | `results_pooled_transformer.csv`, `results_pooled_mlp.csv`, `evaluation_summary_arch.txt`, plot | ~2.4 h |

Seeds (42), the split procedure, the pair list and every hyperparameter are fixed inside the
notebooks; the CSVs the report quotes are committed under `results/`. The `*-v2` / `tawos-arch-ablation`
kernels are the re-runs of the corrected code — the original `tawos-cnn-demo` / `tawos-all-pairs` /
`tawos-pooled` kernels are left untouched on Kaggle so the pre-fix numbers stay available.
