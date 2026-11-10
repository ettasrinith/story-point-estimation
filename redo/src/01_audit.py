# 01_audit.py
#
# CPU-only Kaggle kernel. Produces the *measured* context for the comparative report:
#   - what changing the story-point cap from (0,40] to (0,100] does to the corpus
#   - what changing the BERT window from 128 to 256 does to truncation, on the new corpus
#   - the median baseline recomputed under BOTH filters (the committed baseline is (0,40] only)
#   - a non-neural TF-IDF + Ridge reference on the 39 assigned pairs under the new filter
#
# No training on the GPU is done here, so this kernel can run even when both GPU slots are
# occupied. All numbers it writes are real measurements from the attached Kaggle inputs.

# %% [markdown]
# # 01 — Corpus, baseline and truncation audit (CPU only)
#
# The comparative report needs to separate **two changes** that were made at once in the redo:
# the BERT window (128 → 256 tokens) and the story-point cap ((0,40] → (0,100]). This kernel
# measures both on the same source data, so the report never has to guess what changed.
#
# It also recomputes the seed-42 median baseline and the 39 assigned pairs under both filters,
# because the committed `baseline_results.csv` describes the old (0,40] corpus and cannot be
# used as a control for the new one.
#
# Outputs are tagged `audit` and are deliberately independent of the GPU experiment kernel.
#
# **Inputs**: dataset `ettasrinithreddy/tawos-extracted`; kernel outputs of
# `tawos-bert-256-a` / `tawos-bert-256-b` (metadata CSVs carry the untruncated BERT token
# lengths; the `.npy` arrays are never read here).

# %%
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

SEED = 42
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

csv_hits = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)
assert csv_hits, "attach the tawos-extracted dataset"
raw = pd.read_csv(csv_hits[0])
raw["storypoint"] = pd.to_numeric(raw["storypoint"], errors="coerce")
raw["project"] = raw["project"].astype(str).str.strip()
raw["text"] = raw["text"].fillna("").astype(str)
print("rows in all_issues_sp.csv:", len(raw))

# %% [markdown]
# ## 1. Corpus audit — (0,40] versus (0,100]
#
# Both filters are applied to the *same* raw extract. Nothing is imputed and nothing is
# dropped silently: every exclusion is counted.

# %%
sp = raw.storypoint.dropna()
audit = dict(
    rows_in_csv=int(len(raw)),
    rows_with_storypoint=int(len(sp)),
    rows_without_storypoint=int(len(raw) - len(sp)),
    dropped_sp_le_0=int((sp <= 0).sum()),
    dropped_sp_gt_40=int((sp > 40).sum()),
    dropped_sp_gt_100=int((sp > 100).sum()),
    kept_0_40=int(((sp > 0) & (sp <= 40)).sum()),
    kept_0_100=int(((sp > 0) & (sp <= 100)).sum()),
    added_by_relaxing_40_to_100=int(((sp > 40) & (sp <= 100)).sum()),
    max_storypoint=float(sp.max()),
)
print(json.dumps(audit, indent=2))
json.dump(audit, open(f"{OUT}/audit_corpus.json", "w"), indent=2)

corpora = {}
for tag, hi in (("sp40", 40), ("sp100", 100)):
    d = raw.dropna(subset=["storypoint"]).copy()
    d = d[(d.storypoint > 0) & (d.storypoint <= hi)].reset_index(drop=True)
    corpora[tag] = d
    s = d.storypoint
    print(f"{tag}: {len(d)} issues / {d.project.nunique()} projects | "
          f"mean {s.mean():.3f} median {s.median():.1f} p90 {s.quantile(.9):.1f} "
          f"p99 {s.quantile(.99):.1f} max {s.max():.1f}")
    assert d.project.nunique() == 39

# per-project impact of the filter
rows = []
for proj, g in raw.dropna(subset=["storypoint"]).groupby("project"):
    s = g.storypoint
    rows.append(dict(project=proj, raw=len(g),
                     kept_0_40=int(((s > 0) & (s <= 40)).sum()),
                     kept_0_100=int(((s > 0) & (s <= 100)).sum()),
                     added=int(((s > 40) & (s <= 100)).sum()),
                     pct_added=round(100 * ((s > 40) & (s <= 100)).sum()
                                     / max(1, ((s > 0) & (s <= 40)).sum()), 2),
                     median_sp_40=float(s[(s > 0) & (s <= 40)].median()),
                     median_sp_100=float(s[(s > 0) & (s <= 100)].median())))
proj_tab = pd.DataFrame(rows).sort_values("added", ascending=False)
proj_tab.to_csv(f"{OUT}/corpus_by_project.csv", index=False)
print(proj_tab.head(8).to_string(index=False))

# story-point histogram, both filters, same bins
bins = np.arange(0, 105, 5)
hist = pd.DataFrame({
    "bin_left": bins[:-1],
    "sp40": np.histogram(corpora["sp40"].storypoint, bins=bins)[0],
    "sp100": np.histogram(corpora["sp100"].storypoint, bins=bins)[0],
})
hist.to_csv(f"{OUT}/storypoint_histogram.csv", index=False)

# %% [markdown]
# ## 2. Truncation audit — 128 versus 256 tokens on the new corpus
#
# `n_tokens` in the embedding metadata is the **untruncated** BERT token length. Measuring the
# same column at both cut-offs gives a real number for how much text the redo recovers, on the
# actual (0,100] corpus, instead of an extrapolation from the old (0,40] audit.

# %%
all_csv = sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
meta_by_key = {os.path.basename(p)[:-4]: p for p in all_csv}


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def resolve(name):
    exact = safe(name)
    if exact in meta_by_key:
        return exact
    near = [k for k in meta_by_key if k.rstrip("_") == exact.rstrip("_")]
    assert len(near) == 1, f"no unique metadata for {name!r}: {near}"
    return near[0]


length_rows = []
for tag, d in corpora.items():
    for proj, g in d.groupby("project"):
        m = pd.read_csv(meta_by_key[resolve(proj)])
        # the metadata was produced for the (0,100] corpus; for the (0,40] view select by
        # issue_id so both corpora are measured on exactly the rows they contain
        ids = set(g.issue_id.astype(str))
        L = m.loc[m.issue_id.astype(str).isin(ids), "n_tokens"].values.astype(np.int64)
        length_rows.append(dict(corpus=tag, project=proj, n=len(L),
                                median=float(np.median(L)),
                                p90=float(np.percentile(L, 90)),
                                p_trunc_128=round(100 * float((L > 128).mean()), 2),
                                p_trunc_256=round(100 * float((L > 256).mean()), 2),
                                p_content_lost_128=round(100 * float(
                                    np.maximum(L - 128, 0).sum() / L.sum()), 2),
                                p_content_lost_256=round(100 * float(
                                    np.maximum(L - 256, 0).sum() / L.sum()), 2)))
trunc_by_proj = pd.DataFrame(length_rows)
trunc_by_proj.to_csv(f"{OUT}/truncation_by_project_audit.csv", index=False)

trunc = {}
for tag in corpora:
    d = corpora[tag]
    per = []
    for proj, g in d.groupby("project"):
        m = pd.read_csv(meta_by_key[resolve(proj)])
        per.append(m.loc[m.issue_id.astype(str).isin(set(g.issue_id.astype(str))),
                         "n_tokens"].values.astype(np.int64))
    L = np.concatenate(per)
    trunc[tag] = dict(
        issues=int(len(L)), median_tokens=float(np.median(L)),
        p90_tokens=float(np.percentile(L, 90)), p99_tokens=float(np.percentile(L, 99)),
        max_tokens=int(L.max()),
        trunc_pct_128=round(100 * float((L > 128).mean()), 3),
        trunc_pct_256=round(100 * float((L > 256).mean()), 3),
        content_lost_at_128_pct=round(100 * float(np.maximum(L - 128, 0).sum() / L.sum()), 3),
        content_lost_at_256_pct=round(100 * float(np.maximum(L - 256, 0).sum() / L.sum()), 3))
json.dump(trunc, open(f"{OUT}/truncation_comparison.json", "w"), indent=2)
print(json.dumps(trunc, indent=2))

# %% [markdown]
# ## 3. Seed-42 80:20 partitions and the median baseline under BOTH filters
#
# Identical construction to `02_baseline_median.py` / `10_main.py`: one
# `np.random.default_rng(42)` walked over projects in sorted order, 20% held out per project.
# The 39 assigned PDF pairs are evaluated under both corpora so the baseline column in the
# master comparison table is never borrowed from a different experiment.

# %%
PAIRS = [
    ("XD", "MESOS"), ("Usergrid", "MXNet"), ("Titanium Mobile Platform", "Usergrid"),
    ("The Titanium SDK", "Alloy Framework"), ("Software Cloud", "Appcelerator Daemon"),
    ("Sawtooth", "Appcelerator Studio"), ("Nexus", "Command-Line Interface"),
    ("MXNet", "The Titanium SDK"), ("Mule APIkit", "Titanium Mobile Platform"),
    ("Mule", "C++ driver (Appcelerator)"), ("Moodle", "Atlassian Software Server"),
    ("MongoDB Core Server", "Bamboo"), ("MESOS", "Clover"),
    ("Lyrasis Dura Cloud", "Confluence Cloud"), ("Lsstcorp Data management", "Confluence Server"),
    ("Jira Server", "Crowd"), ("Jira Cloud", "FishEye"), ("Java driver", "Jira Cloud"),
    ("Indy SDK", "Jira Server"), ("Indy Node", "Software Cloud"),
    ("FishEye", "DotNetNuke Platform"), ("Fabric", "Blockchain Explorer"),
    ("Evergreen", "Fabric"), ("DotNetNuke Platform", "Indy Node"), ("DataCass", "Indy SDK"),
    ("Crowd", "Sawtooth"), ("Confluence Server", "Lsstcorp Data management"),
    ("Confluence Cloud", "Lyrasis Dura Cloud"), ("Compass", "MongoDB Core Server"),
    ("Command-Line Interface", "Compass"), ("Clover", "C++ driver (MongoDB)"),
    ("C++ driver (Appcelerator)", "Evergreen"), ("C++ driver (MongoDB)", "Java driver"),
    ("Blockchain Explorer", "Moodle"), ("Bamboo", "Mule"),
    ("Atlassian Software Server", "Mule APIkit"), ("Appcelerator Studio", "Nexus"),
    ("Appcelerator Daemon", "DataCass"), ("Alloy Framework", "XD"),
]
NAME = {
    "XD": "Spring XD", "MESOS": "Apache Mesos", "Usergrid": "Apache Usergrid", "MXNet": "Apache MXNet",
    "Software Cloud": "Atlassian Software Cloud", "Sawtooth": "Hyperledger Sawtooth",
    "Nexus": "Sonatype Nexus", "Bamboo": "Atlassian Bamboo", "Clover": "Atlassian Clover",
    "Confluence Cloud": "Atlassian Confluence Cloud", "Confluence Server": "Atlassian Confluence Server",
    "Jira Server": "Atlassian Jira Server", "Crowd": "Atlassian Crowd",
    "Jira Cloud": "Atlassian Jira Cloud", "FishEye": "Atlassian FishEye",
    "Java driver": "The Mongo Java driver", "Indy SDK": "Hyperledger Indy SDK",
    "Indy Node": "Hyperledger Indy Node", "Fabric": "Hyperledger Fabric",
    "Blockchain Explorer": "Hyperledger Blockchain Explorer", "DataCass": "Spring DataCass",
    "Compass": "MongoDB Compass", "C++ driver (MongoDB)": "The Mongo C++ driver",
    "Evergreen": "The MongoDB Engineering", "C++ driver (Appcelerator)": "Aptana Studio",
    "Atlassian Software Server": "Atlassian Atlassian Software Server",
}
dsn = lambda n: NAME.get(n, n)

split_bank = {}
for tag, d in corpora.items():
    rng = np.random.default_rng(SEED)
    splits, frames, medians = {}, {}, {}
    for proj in sorted(d.project.unique()):
        g = d[d.project == proj].reset_index(drop=True)
        frames[proj] = g
        labels = np.array(g.index)
        rank = {lab: i for i, lab in enumerate(labels)}
        rng.shuffle(labels)
        n_test = max(1, int(round(0.2 * len(labels))))
        splits[proj] = {"test": np.array([rank[v] for v in labels[:n_test]]),
                        "train": np.array([rank[v] for v in labels[n_test:]])}
        medians[proj] = float(np.median(g.storypoint.values[splits[proj]["train"]]))
    split_bank[tag] = dict(splits=splits, frames=frames, medians=medians, df=d)

base_rows = []
for tag, bank in split_bank.items():
    fr, sp_, md = bank["frames"], bank["splits"], bank["medians"]
    for s, t in PAIRS:
        src, tgt = dsn(s), dsn(t)
        y = fr[tgt].storypoint.values[sp_[tgt]["test"]]
        err = np.abs(y - md[src])
        base_rows.append(dict(corpus=tag, pdf_source=s, pdf_target=t, source=src, target=tgt,
                              baseline_prediction=md[src], baseline_mae=float(err.mean()),
                              baseline_mdae=float(np.median(err)),
                              n_train=len(sp_[src]["train"]), n_test=len(y)))
base_pairs = pd.DataFrame(base_rows)
base_pairs.to_csv(f"{OUT}/baseline_assigned_pairs_both_corpora.csv", index=False)

# Name-mapping audit against the *actual dataset* (not a derived manifest): every PDF label must
# resolve to a real project, the 39 labels must cover all 39 projects 1-to-1, and each project must
# appear exactly once as a source and once as a target.
_labels = sorted({x for p in PAIRS for x in p})
_dataset_projects = set(split_bank["sp100"]["frames"])
assert set(dsn(x) for x in _labels) == _dataset_projects, "mapping is not a bijection onto the dataset"
assert len(_labels) == 39 == len(_dataset_projects)
json.dump(dict(
    pdf_labels=len(_labels), dataset_projects=len(_dataset_projects),
    bijection_holds=True,
    every_project_source_once=len({dsn(s) for s, _ in PAIRS}) == 39,
    every_project_target_once=len({dsn(t) for _, t in PAIRS}) == 39,
    note="full three-step mapping (exact / normalised / approved remap) is committed as "
         "results/project_name_mapping.csv and produced by tools/verify_project_mapping.py",
), open(f"{OUT}/project_mapping_validation.json", "w"), indent=2)
print("name mapping: 39 PDF labels -> 39 dataset projects, bijection verified")

summary_rows = []
for tag, bank in split_bank.items():
    fr, sp_, md = bank["frames"], bank["splits"], bank["medians"]
    cross, within = [], []
    for a in sorted(fr):
        for b in sorted(fr):
            if a == b:
                continue
            y = fr[b].storypoint.values[sp_[b]["test"]]
            cross.append(np.abs(y - md[a]).mean())
    for a in sorted(fr):
        y = fr[a].storypoint.values[sp_[a]["test"]]
        within.append(np.abs(y - md[a]).mean())
    sub = base_pairs[base_pairs.corpus == tag]
    summary_rows.append(dict(corpus=tag, issues=len(bank["df"]), projects=39,
                             assigned_pairs=len(sub),
                             assigned_pairs_baseline_mae=float(sub.baseline_mae.mean()),
                             assigned_pairs_baseline_mae_std=float(sub.baseline_mae.std()),
                             assigned_pairs_baseline_mdae=float(sub.baseline_mdae.mean()),
                             all_cross_project_mae=float(np.mean(cross)),
                             all_cross_project_std=float(np.std(cross)),
                             within_project_mae=float(np.mean(within)),
                             held_out_issues_39_pairs=int(sub.n_test.sum())))
baseline_comparison = pd.DataFrame(summary_rows)
baseline_comparison.to_csv(f"{OUT}/baseline_comparison.csv", index=False)
print(baseline_comparison.to_string(index=False))

# %% [markdown]
# ## 4. TF-IDF + Ridge reference on the 39 assigned pairs (new corpus)
#
# A non-neural control trained on exactly the same seed-42 (0,100] partitions. It anchors the
# BERT result: if a BERT + CNN model is only as good as a bag-of-words Ridge regressor, the
# report should say so. Five-fold CV on the source pool selects `alpha`; the target test rows
# are never used for selection. Predictions are clipped at 0 so the comparison is fair to the
# Softplus-headed CNN.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

bank = split_bank["sp100"]
d = bank["df"]
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2,
                      sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
Xall = vec.fit_transform(d.text.values)
print("tf-idf matrix:", Xall.shape, "nnz", Xall.nnz)
# row position -> project frame row
pos = {}
for proj, g in bank["frames"].items():
    pos[proj] = np.flatnonzero(d.project.values == proj)
assert sum(len(v) for v in pos.values()) == len(d)

ALPHAS = (0.1, 1.0, 10.0)
rows = []
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    tr, te = bank["splits"][src]["train"], bank["splits"][tgt]["test"]
    Xtr = Xall[pos[src][tr]]
    Xte = Xall[pos[tgt][te]]
    ytr = bank["frames"][src].storypoint.values[tr].astype(np.float64)
    yte = bank["frames"][tgt].storypoint.values[te].astype(np.float64)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {a: [] for a in ALPHAS}
    for a_, b_ in kf.split(tr):
        for alpha in ALPHAS:
            m = Ridge(alpha=alpha).fit(Xtr[a_], ytr[a_])
            p = np.clip(m.predict(Xtr[b_]), 0, None)
            scores[alpha].append(float(np.abs(p - ytr[b_]).mean()))
    best_alpha = min(ALPHAS, key=lambda a: float(np.mean(scores[a])))
    val_mae = float(np.mean(scores[best_alpha]))
    val_std = float(np.std(scores[best_alpha]))
    m = Ridge(alpha=best_alpha).fit(Xtr, ytr)
    pred = np.clip(m.predict(Xte), 0, None)
    err = np.abs(pred - yte)
    berr = np.abs(yte - bank["medians"][src])
    rows.append(dict(pdf_source=s, pdf_target=t, source=src, target=tgt,
                     model="TF-IDF + Ridge", corpus="sp100", alpha=best_alpha,
                     val_mae_mean=val_mae, val_mae_std=val_std,
                     test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                     baseline_mae=float(berr.mean()), baseline_mdae=float(np.median(berr)),
                     n_train=len(tr), n_test=len(te)))
    if (i + 1) % 10 == 0:
        print(f"  tf-idf {i+1}/39 pairs", flush=True)
tf = pd.DataFrame(rows)
tf.to_csv(f"{OUT}/results_pairs_tfidf_sp100.csv", index=False)
tf_summary = dict(model="TF-IDF + Ridge", corpus="sp100", pairs=len(tf),
                  macro_test_mae=float(tf.test_mae.mean()),
                  macro_test_mae_std=float(tf.test_mae.std()),
                  macro_baseline_mae=float(tf.baseline_mae.mean()),
                  mean_test_mdae=float(tf.test_mdae.mean()),
                  mean_baseline_mdae=float(tf.baseline_mdae.mean()),
                  val_mae_mean=float(tf.val_mae_mean.mean()),
                  val_mae_std=float(tf.val_mae_std.mean()),
                  beats_baseline=int((tf.test_mae < tf.baseline_mae).sum()),
                  issue_weighted_mae=float(np.average(tf.test_mae, weights=tf.n_test)))
json.dump(tf_summary, open(f"{OUT}/master_metrics_tfidf_sp100.json", "w"), indent=2)
print(json.dumps(tf_summary, indent=2))

# %% [markdown]
# ## 5. Plots and reproducibility record

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

fig, ax = plt.subplots(figsize=(7.5, 4.2))
w = 2.0
ax.bar(np.arange(len(bins) - 1) - w / 2, hist.sp40, w, label="(0,40] — old", color="#a0aec0")
ax.bar(np.arange(len(bins) - 1) + w / 2, hist.sp100, w, label="(0,100] — redo", color="#2b6cb0")
ax.set_xticks(np.arange(0, len(bins) - 1, 4), [str(bins[j]) for j in range(0, len(bins) - 1, 4)])
ax.set_yscale("log")
ax.set_xlabel("story point (5-point bins)")
ax.set_ylabel("issues (log scale)")
ax.set_title("What the (0,100] cap adds to the corpus")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/corpus_filter_comparison.png")
plt.close()

fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(2)
ax.bar(x - 0.18, [trunc["sp40"]["trunc_pct_128"], trunc["sp100"]["trunc_pct_128"]],
       0.36, label="128 tokens", color="#dd6b20")
ax.bar(x + 0.18, [trunc["sp40"]["trunc_pct_256"], trunc["sp100"]["trunc_pct_256"]],
       0.36, label="256 tokens", color="#38a169")
for i, k in enumerate(("sp40", "sp100")):
    ax.text(i - 0.18, trunc[k]["trunc_pct_128"] + 0.4,
            f"{trunc[k]['trunc_pct_128']:.1f}%", ha="center", fontsize=8)
    ax.text(i + 0.18, trunc[k]["trunc_pct_256"] + 0.4,
            f"{trunc[k]['trunc_pct_256']:.1f}%", ha="center", fontsize=8)
ax.set_xticks(x, ["(0,40] old corpus", "(0,100] redo corpus"])
ax.set_ylabel("% issues truncated")
ax.set_title("Measured BERT truncation at 128 vs 256 tokens")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/truncation_comparison.png")
plt.close()

json.dump({
    "kernel": "ps3-256-audit", "seed": SEED, "gpu": False,
    "dataset": "ettasrinithreddy/tawos-extracted",
    "embedding_metadata": ["madmaxop123/tawos-bert-256-a", "madmaxop123/tawos-bert-256-b"],
    "filters": {"old": "(0,40]", "redo": "(0,100]"},
    "tfidf": {"max_features": 20000, "ngram_range": [1, 2], "min_df": 2,
              "sublinear_tf": True, "alphas": list(ALPHAS), "folds": 5,
              "clip_predictions_at": 0},
    "runtime_minutes": round((time.time() - t0) / 60, 1),
}, open(f"{OUT}/audit_run_metadata.json", "w"), indent=2)
print("DONE in", round((time.time() - t0) / 60, 1), "min | outputs:", sorted(os.listdir(OUT)))
