# PS-3 comparative results

_Generated 2026-11-10 09:40 UTC by `redo/build_comparison.py`. Every value is read from a results file; see `master_comparison.csv` for its exact source._

## Committed run — 128 tokens, story points (0,40]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.496 | 1.450 | 1.872 | &mdash; | — | 12,438 |
| BERT + CNN (assigned PS-3) | PS-3: 1 source → 1 target | 2.486 | 1.303 | 1.681 | 1.809 | 18/39 | 12,438 |
| Pooled multi-source CNN | Extension: 38 sources → 1 target | 2.191 | 1.340 | 1.414 | &mdash; | 24/39 | 12,438 |
| Pooled Transformer head | Extension: 38 sources → 1 target | 2.242 | 1.260 | 1.511 | &mdash; | 23/39 | 12,438 |
| Pooled MLP head | Extension: 38 sources → 1 target | 2.216 | 1.285 | 1.463 | &mdash; | 23/39 | 12,438 |

## Redo — 256 tokens, story points (0,100]

| Configuration | Protocol | Test MAE | s.d. | Test MdAE | 5-fold val MAE | Beats baseline | Test issues |
|---|---|---:|---:|---:|---:|---:|---:|
| Median baseline | Constant baseline (no training) | 2.897 | 2.098 | 1.897 | &mdash; | — | 12,573 |
| BERT + CNN (redo PS-3) | PS-3: 1 source → 1 target | 2.966 | 2.017 | 1.792 | 1.991 | 15/39 | 12,573 |


**How to read these tables**

- **Test MAE** is the mean of the 39 per-pair test MAEs; **s.d.** is the spread across those pairs. **Test MdAE** is the mean of the 39 per-pair median absolute errors.
- **PS-3: 1 source → 1 target** is the assigned protocol (guideline 3.3). **Extension: 38 sources → 1 target** is a separate extra experiment and must not be read as a PS-3 result or listed as a peer of the PS-3 row.
- **5-fold validation MAE** comes only from the source development pool and is reported separately from the single held-out target test, as required by guideline 3.4.
- The two tables use **different corpora** — 62,193 issues at (0,40] and 62,870 at (0,100]. Compare each model with **its own cohort's baseline**; an old-vs-new MAE difference is descriptive, not a controlled comparison of the token window.
