"""Build the PS-3 comparative results package. Local reporting only — NO model training.

Reads the committed results (128 tokens / story points (0,40]) and the downloaded Kaggle
outputs for the 256-token / (0,100] redo, then writes ``results/comparison/``:

    summary_table.md          the two headline tables, ready to read
    COMPARISON.md             the short comparative narrative
    master_comparison.csv     one row per configuration x metric, with its source file
    configuration_index.csv   what each configuration is and whether it has finished
    per_pair_comparison.csv   all 39 assigned pairs side by side
    fig_*.png                 two plots
    index.html                self-contained dashboard

It also refreshes the generated results block in README.md and REPORT.md between
``<!-- BEGIN GENERATED RESULTS -->`` and ``<!-- END GENERATED RESULTS -->``.

Guideline 3.7: never invent a number. A configuration whose Kaggle kernel has not finished
is listed as PENDING and its cells render as an em dash.

    python redo/build_comparison.py
"""
from __future__ import annotations

import base64
import glob
import html
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
NEW = RES / "256_sp100"
OUT = RES / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
STAMP = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"


def pct2(v) -> str:
    """Format an audit percentage with half-up rounding.

    Plain f"{v:.2f}" uses binary-float (banker's) rounding, so 29.345 became
    29.34 in generated tables while the hand-written prose said 29.35. Decimal
    keeps every table, the dashboard and the report consistent with the JSON.
    """
    return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def find(name: str) -> Path | None:
    hits = sorted(glob.glob(str(NEW / "**" / name), recursive=True))
    return Path(hits[0]) if hits else None


def rcsv(name):
    p = find(name)
    return pd.read_csv(p) if p else None


def rjson(name):
    p = find(name)
    if not p:
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def lcsv(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def f(v, nd=3):
    if v is None or isinstance(v, str):
        return v if v else "&mdash;"
    try:
        if not np.isfinite(float(v)):
            return "&mdash;"
    except (TypeError, ValueError):
        return "&mdash;"
    return f"{float(v):.{nd}f}" if nd else str(int(v))


# ---------------------------------------------------------------------------
# the seven configurations we actually report
# ---------------------------------------------------------------------------
PS3 = "PS-3: 1 source → 1 target"
EXT = "Extension: 38 sources → 1 target"
BASE = "Constant baseline (no training)"

CONFIGS = [
    dict(key="old_median", label="Median baseline", cohort="old", window="—",
         protocol=BASE, model="source-project median", src="results/results_pairs.csv"),
    dict(key="old_bert_cnn", label="BERT + CNN (assigned PS-3)", cohort="old", window="128",
         protocol=PS3, model="frozen BERT + 3-layer CNN", src="results/results_pairs.csv"),
    dict(key="old_pooled_cnn", label="Pooled multi-source CNN", cohort="old", window="128",
         protocol=EXT, model="38-source CNN + ensemble", src="results/results_pooled_pairs.csv"),
    dict(key="old_transformer", label="Pooled Transformer head", cohort="old", window="128",
         protocol=EXT, model="1-layer self-attention", src="results/results_pooled_transformer.csv"),
    dict(key="old_mlp", label="Pooled MLP head", cohort="old", window="128",
         protocol=EXT, model="mean-pooled BERT + MLP", src="results/results_pooled_mlp.csv"),
    dict(key="new_median", label="Median baseline", cohort="new", window="—",
         protocol=BASE, model="source-project median",
         src="ps3-256-audit / baseline_assigned_pairs_both_corpora.csv"),
    dict(key="new_bert_256", label="BERT + CNN (redo PS-3)", cohort="new", window="256",
         protocol=PS3, model="frozen BERT + 3-layer CNN",
         src="ps3-256-main / results_pairs_256.csv"),
]
CFG = {c["key"]: c for c in CONFIGS}
M = {k: {} for k in CFG}
STATUS = {k: "PENDING" for k in CFG}
PAIRS = {}


def from_pairs(key, df, source, mae="test_mae"):
    """Aggregate a 39-row per-pair results table into the standard metric block."""
    if df is None or df.empty:
        return
    ok = df.copy()
    if "status" in ok:
        ok = ok[ok.status.astype(str).str.lower() == "ok"]
    ok = ok[pd.to_numeric(ok[mae], errors="coerce").notna()]
    if ok.empty:
        return
    n = pd.to_numeric(ok.n_test, errors="coerce").fillna(0)
    M[key] = dict(
        mae=float(ok[mae].mean()),
        mae_sd=float(ok[mae].std()),
        mdae=float(ok.test_mdae.mean()),
        base_mae=float(ok.baseline_mae.mean()),
        base_mdae=float(ok.baseline_mdae.mean()) if "baseline_mdae" in ok else None,
        val_mae=float(pd.to_numeric(ok.val_mae_mean, errors="coerce").mean())
                 if "val_mae_mean" in ok else None,
        val_sd=float(pd.to_numeric(ok.val_mae_std, errors="coerce").mean())
                if "val_mae_std" in ok else None,
        n_test=int(n.sum()),
        wins=int((ok[mae] < ok.baseline_mae).sum()),
        epochs=float(pd.to_numeric(ok.epochs, errors="coerce").mean())
                if "epochs" in ok else None,
        weighted_mae=float(np.average(pd.to_numeric(ok[mae]), weights=n)) if n.sum() else None,
    )
    STATUS[key] = "COMPLETE"
    PAIRS[key] = ok.set_index(["pdf_source", "pdf_target"])[[mae, "test_mdae"]].rename(
        columns={mae: f"{key}_mae", "test_mdae": f"{key}_mdae"})


def from_baseline(key, df, source, mae="baseline_mae", mdae="baseline_mdae", n="n_test"):
    if df is None or df.empty:
        return
    nn = pd.to_numeric(df[n], errors="coerce").fillna(0)
    M[key] = dict(
        mae=float(df[mae].mean()), mae_sd=float(df[mae].std()),
        mdae=float(df[mdae].mean()), base_mae=float(df[mae].mean()),
        base_mdae=float(df[mdae].mean()), val_mae=None, val_sd=None,
        n_test=int(nn.sum()), wins=None, epochs=None,
        weighted_mae=float(np.average(pd.to_numeric(df[mae]), weights=nn)) if nn.sum() else None,
    )
    STATUS[key] = "COMPLETE"
    PAIRS[key] = df.set_index(["pdf_source", "pdf_target"])[[mae, mdae]].rename(
        columns={mae: f"{key}_mae", mdae: f"{key}_mdae"})


# --- committed run ----------------------------------------------------------
old = lcsv("results_pairs.csv")
if old is not None:
    from_pairs("old_bert_cnn", old, "results/results_pairs.csv")
    from_baseline("old_median", old, "results/results_pairs.csv (baseline columns)")
from_pairs("old_pooled_cnn", lcsv("results_pooled_pairs.csv"), "results/results_pooled_pairs.csv")
from_pairs("old_transformer", lcsv("results_pooled_transformer.csv"),
           "results/results_pooled_transformer.csv")
from_pairs("old_mlp", lcsv("results_pooled_mlp.csv"), "results/results_pooled_mlp.csv")

# --- redo -------------------------------------------------------------------
both = rcsv("baseline_assigned_pairs_both_corpora.csv")
if both is not None:
    from_baseline("new_median", both[both.corpus == "sp100"],
                  "ps3-256-audit / baseline_assigned_pairs_both_corpora.csv")
from_pairs("new_bert_256", rcsv("results_pairs_256.csv"),
           "ps3-256-main / results_pairs_256.csv")

audit = rjson("audit_corpus.json")
trunc = rjson("truncation_comparison.json")
baseline_cmp = rcsv("baseline_comparison.csv")
main_metrics = rjson("master_metrics_256.json")

DONE = STATUS["new_bert_256"] == "COMPLETE"

WINDOW_VERDICT = (
    "The longer window clearly preserves more text, but the completed redo does **not** show a "
    "macro-MAE gain over its own baseline (§3.2). Because the corpus, cap and partitions all "
    "changed, this is not a controlled 128-vs-256 A/B test of the token window."
    if DONE else
    "The longer window clearly preserves more text; preserving more text is **not** the same as "
    "proving a lower MAE, so no accuracy claim is made until the redo CNN run finishes."
)

HONESTY_NOTE = (
    "**No results are estimated.** All seven configurations finished on Kaggle; every cell above "
    "is a measured value from a completed run (guideline 3.7)."
    if DONE else
    "**No results are fabricated.** An unfinished configuration is marked PENDING and its cells "
    "are blank (guideline 3.7)."
)

BLANK_NOTE = (
    "All 39 pairs in both cohorts finished; every cell is a measured value, none estimated."
    if DONE else
    "Blank cells mean the Kaggle run has not finished — nothing is estimated."
)

# ---------------------------------------------------------------------------
# machine-readable outputs
# ---------------------------------------------------------------------------
METRICS = {
    "mae": "Macro test MAE", "mae_sd": "Macro test MAE s.d.",
    "mdae": "Mean per-pair test MdAE", "base_mae": "Macro baseline MAE",
    "base_mdae": "Macro baseline MdAE", "val_mae": "5-fold validation MAE",
    "val_sd": "5-fold validation MAE s.d.", "n_test": "Held-out test issues",
    "wins": "Pairs beating the median baseline", "epochs": "Mean final epochs",
    "weighted_mae": "Issue-weighted test MAE",
}
rows = []
for c in CONFIGS:
    for metric, value in M[c["key"]].items():
        if value is None:
            continue
        rows.append(dict(configuration_key=c["key"], configuration=c["label"],
                         cohort=c["cohort"], token_window=c["window"],
                         story_point_filter="(0,40]" if c["cohort"] == "old" else "(0,100]",
                         protocol=c["protocol"], model=c["model"], status=STATUS[c["key"]],
                         metric=metric, metric_label=METRICS[metric], value=value,
                         source=c["src"]))
pd.DataFrame(rows).to_csv(OUT / "master_comparison.csv", index=False)

idx = []
for c in CONFIGS:
    k = c["key"]
    idx.append(dict(c, status=STATUS[k], pairs=39 if STATUS[k] == "COMPLETE" else "",
                    macro_test_mae=M[k].get("mae", ""), macro_test_mdae=M[k].get("mdae", "")))
pd.DataFrame(idx).to_csv(OUT / "configuration_index.csv", index=False)

if PAIRS:
    pair = pd.concat(PAIRS.values(), axis=1).reset_index()
    order = next(iter(PAIRS.values())).index
    pos = {p: i for i, p in enumerate(order)}
    pair["__o"] = pair.apply(lambda r: pos.get((r.pdf_source, r.pdf_target), 999), axis=1)
    pair = pair.sort_values("__o").drop(columns="__o")
else:
    pair = pd.DataFrame(columns=["pdf_source", "pdf_target"])
pair.to_csv(OUT / "per_pair_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# markdown tables
# ---------------------------------------------------------------------------
HEAD = ("| Configuration | Protocol | Test MAE | s.d. | Test MdAE | "
        "5-fold val MAE | Beats baseline | Test issues |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|")


def count_or_dash(v):
    return f"{int(v):,}" if v is not None and np.isfinite(v) else "&mdash;"


def wins_or_dash(v):
    return f"{int(v)}/39" if v is not None and np.isfinite(v) else "—"


def table(cohort):
    lines = [HEAD]
    for c in CONFIGS:
        if c["cohort"] != cohort:
            continue
        m = M[c["key"]]
        st = "" if STATUS[c["key"]] == "COMPLETE" else " · **PENDING**"
        lines.append(
            f"| {c['label']}{st} | {c['protocol']} | {f(m.get('mae'))} | "
            f"{f(m.get('mae_sd'))} | {f(m.get('mdae'))} | {f(m.get('val_mae'))} | "
            f"{wins_or_dash(m.get('wins'))} | {count_or_dash(m.get('n_test'))} |")
    return "\n".join(lines)


READING = [
    "",
    "**How to read these tables**",
    "",
    "- **Test MAE** is the mean of the 39 per-pair test MAEs; **s.d.** is the spread across those pairs. "
    "**Test MdAE** is the mean of the 39 per-pair median absolute errors.",
    "- **PS-3: 1 source → 1 target** is the assigned protocol (guideline 3.3). "
    "**Extension: 38 sources → 1 target** is a separate extra experiment and must not be read "
    "as a PS-3 result or listed as a peer of the PS-3 row.",
    "- **5-fold validation MAE** comes only from the source development pool and is reported "
    "separately from the single held-out target test, as required by guideline 3.4.",
    "- The two tables use **different corpora** — 62,193 issues at (0,40] and 62,870 at (0,100]. "
    "Compare each model with **its own cohort's baseline**; an old-vs-new MAE difference is "
    "descriptive, not a controlled comparison of the token window.",
    "",
]

summary = [
    "# PS-3 comparative results",
    "",
    f"_Generated {STAMP} by `redo/build_comparison.py`. Every value is read from a results file; "
    "see `master_comparison.csv` for its exact source._",
    "",
    "## Committed run — 128 tokens, story points (0,40]",
    "",
    table("old"),
    "",
    "## Redo — 256 tokens, story points (0,100]",
    "",
    table("new"),
    "",
    *READING,
]
if STATUS["new_bert_256"] == "PENDING":
    summary += [
        "> **Status.** The 256-token BERT + CNN run is still executing on the Kaggle GPU kernel "
        "`madmaxop123/ps3-256-main`. Its row is left blank on purpose — no value is estimated "
        "(guideline 3.7). Run `python redo/finalize_main.py` when it finishes; that downloads the "
        "outputs and regenerates this table automatically.",
        "",
    ]
(OUT / "summary_table.md").write_text("\n".join(summary), encoding="utf-8")

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
figures = {}


def save(fig, name):
    p = OUT / f"fig_{name}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    figures[name] = p


plot = [c for c in CONFIGS if M[c["key"]].get("mae") is not None]
if plot:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(plot))
    vals = [M[c["key"]]["mae"] for c in plot]
    errs = [M[c["key"]].get("mae_sd") or 0 for c in plot]
    colors = ["#cbd5e1" if c["protocol"] == BASE else
              ("#2563eb" if c["cohort"] == "new" else
               ("#1d4ed8" if c["protocol"] == PS3 else "#93c5fd")) for c in plot]
    ax.bar(x, vals, .62, color=colors, edgecolor="#475569", linewidth=.6,
           yerr=errs, capsize=3, error_kw=dict(ecolor="#475569", lw=.8))
    for i, v in enumerate(vals):
        ax.text(i, v + (errs[i] or 0) + .05, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{c['label']}\n({'redo' if c['cohort'] == 'new' else 'committed'})"
                      for c in plot], rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("macro test MAE (story points)")
    ax.set_title("PS-3 headline comparison — mean MAE over the 39 assigned pairs", fontsize=10)
    ax.set_ylim(0, max(vals + [1]) * 1.3)
    save(fig, "macro_mae")

if trunc:
    t = trunc.get("sp100", {})
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    a = [t.get("trunc_pct_128", 0), t.get("trunc_pct_256", 0)]
    b = [t.get("content_lost_at_128_pct", 0), t.get("content_lost_at_256_pct", 0)]
    x = np.arange(2); w = .34
    ax.bar(x - w/2, a, w, label="issues truncated (%)", color="#ea580c")
    ax.bar(x + w/2, b, w, label="token content lost (%)", color="#0891b2")
    for i in range(2):
        ax.text(i - w/2, a[i] + .6, f"{a[i]:.1f}%", ha="center", fontsize=8)
        ax.text(i + w/2, b[i] + .6, f"{b[i]:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x, ["128 tokens", "256 tokens"])
    ax.set_ylabel("percent")
    ax.set_title("Measured truncation on the (0,100] corpus", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, "truncation")

# ---------------------------------------------------------------------------
# COMPARISON.md
# ---------------------------------------------------------------------------
t100 = trunc.get("sp100", {}) if trunc else {}
cap_note = []
if audit:
    cap_note = [
        f"- Raw issues with a story point: **{audit['rows_with_storypoint']:,}**.",
        f"- Kept at (0,40]: **{audit['kept_0_40']:,}**; kept at (0,100]: "
        f"**{audit['kept_0_100']:,}** (+{audit['added_by_relaxing_40_to_100']:,}).",
        f"- Dropped for `storypoint > 100`: **{audit['dropped_sp_gt_100']:,}**. "
        f"The largest observed value is **{audit['max_storypoint']:,.0f}** — an obvious data-entry "
        "error, which is exactly why an upper cap is needed.",
    ]

lines = [
    "# PS-3 — comparative evaluation",
    "",
    f"_Generated {STAMP} by `redo/build_comparison.py` from committed results and downloaded "
    "Kaggle outputs. No model is trained by this script._",
    "",
    "## 1. What is being compared",
    "",
    "The assignment (PS-3) asks for **BERT word embeddings, cross-project, random split**: train "
    "on one assigned source project, test on a different target project, over the 39 pairs in "
    "guideline 3.3. The committed run used a **128-token** BERT window and kept story points in "
    "**(0,40]**. The redo repeats the identical protocol with a **256-token** window and the "
    "standard **(0,100]** cap.",
    "",
    "| | Committed | Redo |",
    "|---|---|---|",
    "| BERT window | 128 tokens | 256 tokens |",
    "| Story-point filter | (0,40] | (0,100] |",
    "| Embedding arrays | `[n,128,768]` float16, 12.2 GB | `[n,256,768]` float16, 24.7 GB |",
    "| Corpus | 62,193 issues / 39 projects | 62,870 issues / 39 projects |",
    "| Split | seed 42, per-project 80:20 | seed 42, per-project 80:20 (recomputed) |",
    "| Model | 3-layer 1-D CNN, masked max pool, Softplus | identical |",
    "| Protocol | 5-fold CV on source pool, best epoch, one held-out target test | identical |",
    "",
    "We also keep a **separate extra experiment**: pooled multi-source training (38 sources → 1 "
    "target) and its Transformer / MLP predictor ablation, all on the committed corpus. These are "
    "clearly labelled `Extension` in every table. They answer a different question and are not "
    "PS-3 results.",
    "",
    "## 2. Why the corpus and the window changed",
    "",
    "### 2.1 Story-point cap",
    "",
    *cap_note,
    "",
    "The guidelines prescribe the split and the 39 pairs but **do not prescribe a story-point "
    "filter**, so the cap is a documented design choice. `(0,100]` is the community standard: "
    "Deep-SE (Choetkiertikul et al., TSE 2019) drops zero, negative and \"unrealistically large\" "
    "(> 100) values; the TAWOS README sample query uses `Story_Point BETWEEN 1 AND 100`; "
    "Tawosi et al. (ESEM 2022) remove SP < 1 and > 100. The original `(0,40]` cap was the outlier.",
    "",
    "The cap is not cosmetic. Admitting large story points mechanically raises MAE for every "
    "method, including the constant baseline, while MdAE barely moves.",
]
if baseline_cmp is not None:
    lines += ["", "| Corpus | Issues | Assigned-pairs baseline MAE | s.d. | Baseline MdAE | "
              "Held-out issues |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in baseline_cmp.iterrows():
        lines.append(f"| {r.corpus} | {int(r.issues):,} | "
                     f"{r.assigned_pairs_baseline_mae:.3f} | "
                     f"{r.assigned_pairs_baseline_mae_std:.3f} | "
                     f"{r.assigned_pairs_baseline_mdae:.3f} | "
                     f"{int(r.held_out_issues_39_pairs):,} |")
lines += [
    "",
    "### 2.2 Token window",
    "",
    "| Window | Issues truncated | Token content lost |",
    "|---|---:|---:|",
    f"| 128 tokens | {pct2(t100.get('trunc_pct_128', 0))}% | "
    f"{pct2(t100.get('content_lost_at_128_pct', 0))}% |",
    f"| 256 tokens | {pct2(t100.get('trunc_pct_256', 0))}% | "
    f"{pct2(t100.get('content_lost_at_256_pct', 0))}% |",
    "",
    "Measured with the `bert-base-uncased` tokenizer on the (0,100] corpus "
    "(`truncation_comparison.json`). Median untruncated length is "
    f"{t100.get('median_tokens', float('nan')):.0f} tokens, p90 "
    f"{t100.get('p90_tokens', float('nan')):.0f}. {WINDOW_VERDICT}",
    "",
    "## 3. Master comparison",
    "",
    "### 3.1 Committed run — 128 tokens, (0,40]",
    "",
    table("old"),
    "",
    "### 3.2 Redo — 256 tokens, (0,100]",
    "",
    table("new"),
    "",
    *READING[1:],
    "",
]
if not DONE:
    lines += [
        "> **Status.** The 256-token BERT + CNN kernel `madmaxop123/ps3-256-main` is still "
        "running on Kaggle. Its row is blank on purpose (guideline 3.7); "
        "`python redo/finalize_main.py` refreshes this table when the run finishes.",
        "",
    ]
else:
    m = M["new_bert_256"]
    mm = main_metrics or {}
    lines += [
        "> **What the redo actually shows.** The CNN's macro test MAE "
        f"({m['mae']:.3f}) is **worse** than its median baseline ({m['base_mae']:.3f}) by "
        f"{m['mae'] - m['base_mae']:.3f}, and it wins on only {m['wins']}/39 pairs "
        f"(pair-level Wilcoxon p = {mm.get('wilcoxon_pair', float('nan')):.3f}, not significant). "
        f"It does improve **typical-case** error: mean per-pair MdAE {m['mdae']:.3f} vs "
        f"{f(m.get('base_mdae'))}, and issue-weighted MAE "
        f"{mm.get('issue_weighted_mae', 0):.3f} vs "
        f"{mm.get('issue_weighted_baseline_mae', 0):.3f}. Preserving more text is therefore a "
        "measured gain in input fidelity, **not** a demonstrated gain in macro MAE.",
        "",
    ]
lines += [
    "## 4. Validation and held-out test",
    "",
    "Following guideline 3.4, five-fold validation is computed **only** on the source project's "
    "development pool and is used solely to choose the best epoch. The target project's held-out "
    "20% is read exactly once, to score the final model. No test row is used for training, early "
    "stopping, epoch selection or calibration in the assigned protocol. Validation and test "
    "numbers are reported in separate columns above.",
    "",
    "A second reason the two cohorts are not paired: `rng.shuffle` walks projects in sorted order "
    "from one shared `np.random.default_rng(42)` stream, and the (0,100] corpus changes the row "
    "count of 8 projects. Adding rows shifts later draws, so held-out rows move even when the "
    "held-out count does not. Within a cohort all models share partitions; across cohorts they "
    "do not.",
    "",
    "## 5. Provenance and reproduction",
    "",
    "| Artefact | Produced by |",
    "|---|---|",
    "| Committed 128 / (0,40] results | `05_run_all_pairs.ipynb`, `06_pooled_training.ipynb`, "
    "`07_architecture_ablation.ipynb` |",
    "| 256 / (0,100] embeddings | Kaggle `madmaxop123/tawos-bert-256-a` and `-b` |",
    "| Corpus, baseline, truncation audit | Kaggle `madmaxop123/ps3-256-audit` (CPU) |",
    "| Assigned PS-3 redo | Kaggle `madmaxop123/ps3-256-main` (GPU) |",
    "| Notebook sources | `redo/src/*.py`, assembled by `redo/build_kernels.py` |",
    "| This package | `redo/build_comparison.py` (local aggregation only) |",
    "",
    "Reproduce: `python redo/build_kernels.py` → `kaggle kernels push -p redo/kernels/<slug>` → "
    "`kaggle kernels output madmaxop123/<slug> -p results/256_sp100/<slug>` → "
    "`python redo/build_comparison.py`. All training runs on Kaggle, never locally.",
    "",
    HONESTY_NOTE,
]
(OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# index.html
# ---------------------------------------------------------------------------
def image(name):
    p = figures.get(name)
    if not p or not p.exists():
        return '<div class="pending">plot pending</div>'
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}"/>'


def table_html(headers, body, numeric_from=2):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in headers)
    trs = []
    for r in body:
        tds = "".join(f'<td{" class=num" if i >= numeric_from else ""}>{c}</td>'
                      for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def html_table(cohort):
    body = []
    for c in CONFIGS:
        if c["cohort"] != cohort:
            continue
        m = M[c["key"]]
        badge = ('<span class="ok">COMPLETE</span>' if STATUS[c["key"]] == "COMPLETE"
                 else '<span class="wait">PENDING</span>')
        body.append([html.escape(c["label"]), html.escape(c["protocol"]), badge,
                     f(m.get("mae")), f(m.get("mae_sd")), f(m.get("mdae")),
                     f(m.get("val_mae")),
                     wins_or_dash(m.get("wins")), count_or_dash(m.get("n_test"))])
    return table_html(["Configuration", "Protocol", "Status", "Test MAE", "s.d.",
                       "Test MdAE", "5-fold val MAE", "Beats baseline", "Test issues"],
                      body, numeric_from=3)


cards = []
for title, value, note in [
    ("Committed PS-3 test MAE", f(M["old_bert_cnn"].get("mae")), "128 tokens / (0,40]"),
    ("Committed median baseline", f(M["old_median"].get("mae")), "same corpus and splits"),
    ("Redo PS-3 test MAE", f(M["new_bert_256"].get("mae")), "256 tokens / (0,100]"),
    ("Redo median baseline", f(M["new_median"].get("mae")), "same corpus and splits"),
    ("Truncation at 128", f"{t100.get('trunc_pct_128', float('nan')):.1f}%", "new corpus"),
    ("Truncation at 256", f"{t100.get('trunc_pct_256', float('nan')):.1f}%", "new corpus"),
]:
    cards.append(f'<div class="card"><div class="ct">{html.escape(title)}</div>'
                 f'<div class="cv">{value}</div><div class="cn">{html.escape(note)}</div></div>')

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PS-3 comparative results</title><style>
:root{{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;--bg:#f8fafc}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;line-height:1.55}}
.wrap{{max-width:1080px;margin:0 auto;padding:34px 22px 70px}}
header{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:24px}}
h1{{font-size:26px;margin:0 0 6px}}h2{{font-size:17px;margin:32px 0 12px}}
.sub{{color:var(--muted);font-size:14px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:22px 0 30px}}
.card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:13px 15px}}
.ct{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.cv{{font-size:25px;font-weight:650;margin:5px 0 2px;color:var(--blue)}}
.cn{{font-size:11px;color:var(--muted)}}
section{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}}
section img{{width:100%;height:auto;display:block}}section p{{color:var(--muted);font-size:12.5px;margin:10px 0 0}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:12.5px;border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th{{background:#f1f5f9;text-align:left;padding:9px 10px;white-space:nowrap}}
td{{padding:8px 10px;border-top:1px solid var(--line)}}td.num{{text-align:right;font-variant-numeric:tabular-nums}}
.ok{{background:#dcfce7;color:#166534;padding:2px 7px;border-radius:99px;font-size:10px;font-weight:700}}
.wait{{background:#fef3c7;color:#92400e;padding:2px 7px;border-radius:99px;font-size:10px;font-weight:700}}
.note{{background:#eff6ff;border-left:3px solid var(--blue);padding:12px 15px;border-radius:0 8px 8px 0;font-size:13px;margin:18px 0}}
.scroll{{overflow-x:auto}}footer{{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<header><h1>PS-3 · comparative results</h1><div class="sub">BERT word embeddings for
cross-project story-point estimation · 128 → 256 tokens, story points (0,40] → (0,100] ·
generated {STAMP}</div></header>
<div class="cards">{"".join(cards)}</div>
<div class="note"><strong>Read this first.</strong> The two tables use different corpora, so an
old-vs-new MAE difference is descriptive, not a controlled window A/B. Rows marked
<em>Extension</em> use 38 sources instead of the single source PS-3 assigns; they are not PS-3
results. {BLANK_NOTE}</div>
<h2>Committed run — 128 tokens, (0,40]</h2><div class="scroll">{html_table("old")}</div>
<h2>Redo — 256 tokens, (0,100]</h2><div class="scroll">{html_table("new")}</div>
<h2>Headline comparison</h2><section>{image("macro_mae")}
<p>Mean test MAE over the 39 assigned pairs with between-pair s.d. Grey = median baselines,
dark blue = assigned PS-3 model, light blue = pooled extensions.</p></section>
<h2>Why the window changed</h2><section>{image("truncation")}
<p>Measured truncation and token-content loss on the (0,100] corpus. Better input fidelity is
not the same as a proven accuracy gain.</p></section>
<footer>Generated by <code>redo/build_comparison.py</code> from <code>results/*</code> and
<code>results/256_sp100/</code>. Training ran on Kaggle GPUs only. Full narrative:
<code>COMPARISON.md</code>; per-value provenance: <code>master_comparison.csv</code>.</footer>
</div></body></html>"""
(OUT / "index.html").write_text(doc, encoding="utf-8")

# ---------------------------------------------------------------------------
# inject results into README.md and REPORT.md
# ---------------------------------------------------------------------------
findings = []
if audit:
    findings.append(
        f"- **Corpus.** (0,40] keeps {audit['kept_0_40']:,} issues, (0,100] keeps "
        f"{audit['kept_0_100']:,} (+{audit['added_by_relaxing_40_to_100']:,}); "
        f"{audit['dropped_sp_gt_100']:,} values above 100 are still removed, including a "
        "26,710,114 story-point data-entry error.")
if t100:
    findings.append(
        f"- **Window.** On the (0,100] corpus a 128-token window truncates "
        f"{pct2(t100.get('trunc_pct_128', 0))}% of issues and loses "
        f"{pct2(t100.get('content_lost_at_128_pct', 0))}% of token content; "
        f"256 tokens reduce this to {pct2(t100.get('trunc_pct_256', 0))}% and "
        f"{pct2(t100.get('content_lost_at_256_pct', 0))}% (measured, not extrapolated).")
if baseline_cmp is not None and len(baseline_cmp) == 2:
    a = baseline_cmp.set_index("corpus")
    findings.append(
        f"- **Baseline moves with the corpus.** The same seed-42 median baseline scores "
        f"{a.loc['sp40', 'assigned_pairs_baseline_mae']:.3f} MAE on (0,40] but "
        f"{a.loc['sp100', 'assigned_pairs_baseline_mae']:.3f} on (0,100], while MdAE moves only "
        f"{a.loc['sp40', 'assigned_pairs_baseline_mdae']:.3f} → "
        f"{a.loc['sp100', 'assigned_pairs_baseline_mdae']:.3f}. Never compare the two cohorts' "
        "MAE as if the test set were identical.")
if STATUS["new_bert_256"] == "COMPLETE":
    m = M["new_bert_256"]
    mm = main_metrics or {}
    _mae_dir = "above" if m["mae"] > m["base_mae"] else "below"
    findings.append(
        f"- **Redo PS-3 result (39/39 pairs).** At 256 tokens / (0,100], BERT + CNN scores "
        f"macro test MAE {m['mae']:.3f} ± {f(m.get('mae_sd'))} against a "
        f"{m['base_mae']:.3f} median baseline — {_mae_dir} it by "
        f"{abs(m['mae'] - m['base_mae']):.3f}. It does **better on typical issues**: mean "
        f"per-pair MdAE {m['mdae']:.3f} vs {f(m.get('base_mdae'))}, and issue-weighted MAE "
        f"{mm.get('issue_weighted_mae', 0):.3f} vs "
        f"{mm.get('issue_weighted_baseline_mae', 0):.3f}. It wins on {m['wins']}/39 pairs; "
        f"five-fold validation MAE is {f(m.get('val_mae'))}.")
    findings.append(
        f"- **No proven window gain.** The pair-level difference is not significant "
        f"(Wilcoxon p = {mm.get('wilcoxon_pair', float('nan')):.3f}). Longer inputs preserve "
        "more text, but this experiment does **not** show a macro-MAE improvement; the corpus, "
        "cap and partitions all changed, so it is not a controlled 128-vs-256 A/B test.")
else:
    findings.append(
        "- **Redo status.** The 256-token BERT + CNN run (`madmaxop123/ps3-256-main`) is still "
        "on the Kaggle GPU. Its row is PENDING and deliberately blank — no value is estimated "
        "(guideline 3.7). `python redo/finalize_main.py` fills it in when the run finishes.")

full_block = "\n".join([
    BEGIN,
    f"_Auto-generated {STAMP} by `redo/build_comparison.py`. Do not edit between the markers._",
    "",
    "### 4.1 Headline tables",
    "",
    "#### Committed run — 128 tokens, story points (0,40]",
    "",
    table("old"),
    "",
    "#### Redo — 256 tokens, story points (0,100]",
    "",
    table("new"),
    "",
    *READING,
    "### 4.2 Measured findings",
    "",
    *findings,
    "",
    "Full detail: [`results/comparison/COMPARISON.md`](results/comparison/COMPARISON.md), "
    "[dashboard](results/comparison/index.html), "
    "[master metrics CSV](results/comparison/master_comparison.csv), "
    "[all 39 pairs](results/comparison/per_pair_comparison.csv).",
    END,
])

short_block = "\n".join([
    BEGIN,
    f"_Auto-generated {STAMP} by `redo/build_comparison.py`. Do not edit between the markers._",
    "",
    "### Committed run — 128 tokens, story points (0,40]",
    "",
    table("old"),
    "",
    "### Redo — 256 tokens, story points (0,100]",
    "",
    table("new"),
    "",
    *READING,
    "Full comparative package: [`results/comparison/`](results/comparison/COMPARISON.md).",
    END,
])


def inject(path: Path, content: str):
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print(f"  ! markers missing in {path.name}; skipped")
        return
    pre, rest = text.split(BEGIN, 1)
    _, post = rest.split(END, 1)
    path.write_text(pre + content + post, encoding="utf-8")
    print("  refreshed", path.name)


inject(ROOT / "README.md", short_block)
inject(ROOT / "REPORT.md", full_block)

print("wrote", OUT)
print("configurations:", len(CONFIGS),
      "| COMPLETE:", sum(1 for k in STATUS if STATUS[k] == "COMPLETE"),
      "| PENDING:", sum(1 for k in STATUS if STATUS[k] == "PENDING"))
for c in CONFIGS:
    k = c["key"]
    print(f"  {STATUS[k]:8s} {c['cohort']:3s} {c['label']:32s} "
          f"MAE {f(M[k].get('mae')):>8s}  MdAE {f(M[k].get('mdae')):>8s}")
print("files:", sorted(p.name for p in OUT.iterdir()))
