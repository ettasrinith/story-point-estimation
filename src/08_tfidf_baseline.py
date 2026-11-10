# 08_tfidf_baseline.py
#
# Lexical reference point: TF-IDF + Ridge on the same partitions. CPU only, no embeddings.

# %% [markdown]
# # 08 — TF-IDF + Ridge Lexical Baseline (39 assigned pairs, same partitions)
#
# The median constant is a floor, not the strongest naive reference: published evaluations on this
# dataset family show a lexical model can match or beat frozen-embedding models. The comparison
# therefore includes TF-IDF + Ridge as a strong cheap baseline over the *same* seed-42 partitions,
# the same 39 assigned pairs and the same single held-out test evaluation as 05.
#
# **Protocol per pair (mirrors 05):**
# 1. The Ridge alpha is selected from {0.3, 1, 3, 10} by 5-fold cross-validation on the source's
#    development pool — development data only, like every other selection in this repo. The
#    vectorizer is refit inside each fold: the vocabulary is part of the model, so fitting it on
#    the full pool would hand the validation fold's words to training.
# 2. The final model (vectorizer + Ridge at the chosen alpha) trains on the full pool.
# 3. One evaluation on the target's held-out 20%. Predictions are clipped at 0 — the same
#    positivity the CNN's Softplus guarantees.
#
# A **pooled** variant (train on the other 38 projects' development pools, alpha selected by 5-fold
# CV on that pool) answers the same question as notebook 06 for a lexical model and is reported
# separately, like all pooled results in this repo.
#
# **Inputs:** dataset `tawos-extracted` (`all_issues_sp.csv`, `baseline_results.csv`).
# **CPU only** (~20–30 min). **Outputs:** `results_tfidf_pairs.csv`, `results_tfidf_pooled.csv`,
# `tfidf_summary.txt`, `predictions_tfidf/<source>__<target>.csv`.

# %%
import glob
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

SEED = 42
SP_MIN, SP_MAX = 0, 40
N_FOLDS = 5
ALPHAS = (0.3, 1.0, 3.0, 10.0)

csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()

# Seed-42 80:20 split per project, rebuilt identically to 02/05/06 and verified the same way:
# every source's train-pool median must equal the constant the baseline predicted from that pool.
rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    labels = np.array(g.index)                  # copy: safe to shuffle in place
    rank = {lab: i for i, lab in enumerate(labels)}
    rng.shuffle(labels)
    n_test = max(1, int(round(0.2 * len(labels))))
    splits[proj] = {"test": np.array([rank[v] for v in labels[:n_test]]),
                    "train": np.array([rank[v] for v in labels[n_test:]])}

base = pd.read_csv(glob.glob("/kaggle/input/**/baseline_results.csv", recursive=True)[0])
base["source"] = base["source"].astype(str).str.strip()
base["target"] = base["target"].astype(str).str.strip()
within = base[base.target.str.contains("WITHIN")].set_index("source")
for proj in splits:
    pool = df[df.project == proj].storypoint.values[splits[proj]["train"]]
    med = float(np.median(pool))
    assert abs(med - float(within.loc[proj, "pred"])) < 1e-6, (proj, med, within.loc[proj, "pred"])
print(f"{len(df)} issues | {df.project.nunique()} projects | splits reproduce the baseline "
      "partitions (verified by train-pool medians)")

# %% [markdown]
# ## Model
#
# Word unigrams+bigrams, sublinear TF, `min_df=2`, at most 200k features; Ridge on the L2-scale
# target. All settings are fixed here (see `configs/08_tfidf_baseline.json`) and selected-alpha
# statistics are recorded per pair so the selection is auditable.

# %%
def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200_000)


def select_alpha(texts, y, seed=SEED):
    """5-fold CV on the development pool; returns (best_alpha, mean val MAE of every alpha)."""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = {a: [] for a in ALPHAS}
    for a_idx, b_idx in kf.split(np.arange(len(y))):
        vec = make_vectorizer()
        xa = vec.fit_transform(texts[a_idx])
        xb = vec.transform(texts[b_idx])
    for a in ALPHAS:
        mdl = Ridge(alpha=a).fit(xa, y[a_idx])
        scores[a].append(float(np.abs(mdl.predict(xb) - y[b_idx]).mean()))
    means = {a: float(np.mean(v)) for a, v in scores.items()}
    return min(means, key=means.get), means


def fit_predict(texts_tr, y_tr, texts_te, alpha):
    vec = make_vectorizer()
    mdl = Ridge(alpha=alpha).fit(vec.fit_transform(texts_tr), y_tr)
    return np.maximum(mdl.predict(vec.transform(texts_te)), 0.0)


# %% [markdown]
# ## The 39 assigned pairs
#
# Same pair list and name mapping as 05/06 (two instructor-approved mappings).

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
    "The Titanium SDK": "The Titanium SDK", "Software Cloud": "Atlassian Software Cloud",
    "Sawtooth": "Hyperledger Sawtooth", "Nexus": "Sonatype Nexus", "Bamboo": "Atlassian Bamboo",
    "Clover": "Atlassian Clover", "Confluence Cloud": "Atlassian Confluence Cloud",
    "Confluence Server": "Atlassian Confluence Server", "Jira Server": "Atlassian Jira Server",
    "Crowd": "Atlassian Crowd", "Jira Cloud": "Atlassian Jira Cloud", "FishEye": "Atlassian FishEye",
    "Java driver": "The Mongo Java driver", "Indy SDK": "Hyperledger Indy SDK",
    "Indy Node": "Hyperledger Indy Node", "Fabric": "Hyperledger Fabric",
    "Blockchain Explorer": "Hyperledger Blockchain Explorer", "DataCass": "Spring DataCass",
    "Compass": "MongoDB Compass", "C++ driver (MongoDB)": "The Mongo C++ driver",
    "Evergreen": "The MongoDB Engineering", "C++ driver (Appcelerator)": "Aptana Studio",
    "Atlassian Software Server": "Atlassian Atlassian Software Server",
}


def dsn(pdf_name):
    return NAME.get(pdf_name, pdf_name)


assert len(PAIRS) == 39 and len(set(PAIRS)) == 39


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


# %%
os.makedirs("/kaggle/working/predictions_tfidf", exist_ok=True)
base_idx = base.set_index(["source", "target"])
rows, t0 = [], time.time()
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    tr_pos, te_pos = splits[src]["train"], splits[tgt]["test"]
    g_src = df[df.project == src].reset_index(drop=True)
    g_tgt = df[df.project == tgt].reset_index(drop=True)
    texts_tr = g_src.text.values[tr_pos]
    y_tr = g_src.storypoint.values[tr_pos].astype(np.float32)
    texts_te = g_tgt.text.values[te_pos]
    y_te = g_tgt.storypoint.values[te_pos].astype(np.float32)

    alpha, alpha_means = select_alpha(texts_tr, y_tr)
    pred = fit_predict(texts_tr, y_tr, texts_te, alpha)
    err = np.abs(pred - y_te)

    pred_const = float(base_idx.loc[(src, tgt)].pred)
    pd.DataFrame({
        "source_project": src, "target_project": tgt,
        "issue_id": g_tgt.issue_id.values[te_pos], "issue_key": g_tgt.issue_key.values[te_pos],
        "actual_story_point": y_te, "predicted_story_point": pred, "absolute_error": err,
        "source_median_prediction": pred_const,
    }).to_csv(f"/kaggle/working/predictions_tfidf/{safe(src)}__{safe(tgt)}.csv", index=False)

    brow = base_idx.loc[(src, tgt)]
    rows.append(dict(pdf_source=s, pdf_target=t, source=src, target=tgt, alpha=alpha,
                     alpha_cv_means=";".join(f"{a}:{alpha_means[a]:.3f}" for a in ALPHAS),
                     test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                     n_test=int(len(y_te)),
                     baseline_mae=float(brow.mae), baseline_mdae=float(brow.mdae),
                     beats_baseline=bool(err.mean() < float(brow.mae))))
    pd.DataFrame(rows).to_csv("/kaggle/working/results_tfidf_pairs.csv", index=False)
    print(f"[{i + 1}/39] {s} -> {t}: alpha {alpha} | test MAE {rows[-1]['test_mae']:.3f} "
          f"(baseline {rows[-1]['baseline_mae']:.3f}), MdAE {rows[-1]['test_mdae']:.3f} "
          f"| {time.time() - t0:.0f}s", flush=True)

# %% [markdown]
# ## Pooled variant (the 06 question for a lexical model)
#
# Same protocol as 06 minus the embeddings: for each target, train on the other 38 projects'
# development pools; alpha by 5-fold CV on that pool; one test evaluation on the target's held-out
# 20%. Reported separately from the assigned single-source protocol, as everywhere in this repo.

# %%
names = sorted(df.project.unique())


def project_pool(p):
    """One project's development pool as aligned (texts, story points) arrays."""
    g = df[df.project == p].reset_index(drop=True)
    return g.text.values[splits[p]["train"]], g.storypoint.values[splits[p]["train"]].astype(np.float32)


pool_by_target = {}
for tgt in names:
    parts = [project_pool(p) for p in names if p != tgt]
    pool_by_target[tgt] = (np.concatenate([t for t, _ in parts]),
                           np.concatenate([y for _, y in parts]))

# one alpha for the pooled model, selected on a fixed development sample of the pool (dev only)
texts_all = np.concatenate([project_pool(p)[0] for p in names])
y_all = np.concatenate([project_pool(p)[1] for p in names])
sample = np.random.default_rng(SEED).choice(len(texts_all), 30000, replace=False)
alpha_pooled, pooled_means = select_alpha(texts_all[sample], y_all[sample])
print(f"pooled alpha {alpha_pooled} | CV means {pooled_means}")

rows_p = []
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    te_pos = splits[tgt]["test"]
    g_tgt = df[df.project == tgt].reset_index(drop=True)
    texts_p, y_p = pool_by_target[tgt]
    pred = fit_predict(texts_p, y_p.astype(np.float32), g_tgt.text.values[te_pos], alpha_pooled)
    err = np.abs(pred - g_tgt.storypoint.values[te_pos].astype(np.float32))
    rows_p.append(dict(pdf_source=s, pdf_target=t, source=src, target=tgt, alpha=alpha_pooled,
                       test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                       n_test=int(len(err))))
    pd.DataFrame(rows_p).to_csv("/kaggle/working/results_tfidf_pooled.csv", index=False)
    print(f"[pooled {i + 1}/39] {s} -> {t}: MAE {rows_p[-1]['test_mae']:.3f} "
          f"| {time.time() - t0:.0f}s", flush=True)

# %% [markdown]
# ## Summary
#
# Macro (pair-average) and issue-weighted MAE, mean per-pair MdAE and pooled MdAE, against the
# median baseline on the same partitions. These rows feed the §3.4-style comparative table that
# notebook 10 assembles for every representation.

# %%
res = pd.DataFrame(rows)
res_p = pd.DataFrame(rows_p)
preds_all = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("/kaggle/working/predictions_tfidf/*.csv"))],
                      ignore_index=True)
macro, macro_std = res.test_mae.mean(), res.test_mae.std()
weighted = float(np.average(res.test_mae, weights=res.n_test))
lines = [
    "TF-IDF + Ridge lexical baseline - 39 assigned cross-project pairs (same partitions as 02/05)",
    f"seed {SEED} | SP in (0,40] | 5-fold CV alpha selection on the source pool | predictions clipped at 0",
    "",
    f"macro (pair-average) test MAE : {macro:.3f} +/- {macro_std:.3f}   (baseline {res.baseline_mae.mean():.3f})",
    f"issue-weighted test MAE       : {weighted:.3f}   (baseline "
    f"{float(np.average(res.baseline_mae, weights=res.n_test)):.3f})",
    f"pooled over {len(preds_all)} test issues: MAE {preds_all.absolute_error.mean():.3f} "
    f"| MdAE {preds_all.absolute_error.median():.3f}",
    f"mean of per-pair test MdAE    : {res.test_mdae.mean():.3f}   (baseline {res.baseline_mdae.mean():.3f})",
    f"beats baseline MAE on         : {int(res.beats_baseline.sum())}/{len(res)} pairs",
    "",
    f"POOLED lexical model (other-38 pools, alpha {alpha_pooled}): macro test MAE "
    f"{res_p.test_mae.mean():.3f} +/- {res_p.test_mae.std():.3f} "
    f"| mean per-pair MdAE {res_p.test_mdae.mean():.3f}",
    "pooled rows are the lexical counterpart of notebook 06 and are NOT the assigned protocol",
]
txt = "\n".join(lines)
open("/kaggle/working/tfidf_summary.txt", "w").write(txt)
print(txt)
