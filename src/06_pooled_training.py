# 06_pooled_training.py
#
# Multi-source extension of the PS-3 experiment. Different question from 05 — kept separate on
# purpose, and its numbers are not a drop-in replacement for the assigned per-pair results.

# %% [markdown]
# # 06 — Pooled-Source Training with Calibration (multi-source extension)
#
# **This answers a different question than the assignment.** PS-3 prescribes one source → one target
# (`XD → MESOS`), which is what `05` measures. Here the model for a target trains on the combined
# development pools of *all other 38 projects* (`{everything except MESOS} → MESOS`), GPT2SP-style
# (Fu & Tantithamthavorn, TSE 2023). Both are reported, never merged into one table.
#
# Three things change at once relative to `05`, which is why the ablation-style reporting below
# matters: pooled sources, a `log1p` target, and a 3-seed ensemble.
#
# **What this revision adds (on top of the corrections already listed at the bottom):**
#
# 1. *Target-blind out-of-fold calibration.* Cross-project predictions inherit the training pool's
#    scale. Each target's predictions are affinely calibrated `a·p + b`, where `a` and `b` are fitted
#    on the **out-of-fold predictions of the other 38 projects** from the grouped cross-validation
#    below — the target's own rows (development *and* test) are excluded from the fit, exactly as
#    they are excluded from its model's training. No target data of any kind enters training, epoch
#    selection or calibration; the held-out 20% test rows are only ever read to score. This corrects
#    the systematic underestimation that `expm1(log1p(·))` regression introduces on skewed labels.
# 2. *Protocol note (PS-3 compliance).* Training uses the development pools of the **source
#    projects only** — for a target, that is the other 38 projects. The target's development data is
#    never trained on by its own model, never used to pick its epoch count (folds that held the
#    target out are excluded from that target's average), and never used for calibration. The
#    39 assigned pairs and the seed-42 partitions are unchanged from `05`.
# 3. *Tuned epoch caps.* The corrected CV now explores up to 10 epochs and the final model is capped
#    at 8 (the per-source run's fold best-epochs averaged 8.8; the grouped pooled CV lands lower).
# 4. *Paired significance tests.* Issue-level Wilcoxon signed-rank tests of |model error| against the
#    pooled-median and single-source-median errors, plus per-pair tests across the 39 pairs.
#
# **Fixes carried from the previous revision (kept here so the notebook stands alone)**
#
# 1. *Epoch selection was a no-op.* The first build capped CV at `epochs=6` with patience 4 and every
#    fold ran all six epochs, so `FIXED_EPOCHS = 6` was the ceiling, not a chosen value. Folds now
#    report their true best epoch under `CV_MAX_EPOCHS`.
# 2. *Epoch selection was not target-blind.* Folds are grouped by project (`GroupKFold`) and, for a
#    given target, only folds whose held-out project group excludes that target contribute to its
#    epoch count. No target data configures its own model's epoch count. (Never was leakage of the
#    held-out 20% test sets, which were never involved.)
# 3. *The comparison was unfair.* A **pooled median baseline** — the same 38-project pool, no model —
#    is computed and reported next to the single-source baseline.
# 4. *The split was not the seed-42 one.* Fixed as in `05`; the partition is checked against the
#    baseline's own medians before training.
# 5. The fold metric uses the best epoch rather than the stopping epoch, padding is masked before
#    every convolution, and embeddings are verified against the issue rows.
#
# **Inputs:** dataset `tawos-extracted`, kernel output `tawos-bert-embeddings`. **Needs GPU**
# (~5–7 h on a T4: 39 targets × 3 seeds × ≤8 epochs over a ~48k-row pool, plus grouped CV) and
# ~13 GB RAM for the pooled fp16 array. Results are rewritten after every pair, so an interrupted
# session still leaves a complete CSV of everything finished.
#
# **Note on the committed results:** `results/results_pooled_pairs.csv` (test MAE 2.024) came from the
# *earlier* protocol — positional splits and a 6-epoch ceiling. It is kept as a historical artifact
# and labelled as such in README/REPORT; this notebook's run replaces it.

# %%
import glob
import os
import re
import time

import numpy as np
import pandas as pd

SEED = 42
MAX_LEN = 128
EMB_WIDTH = 768
BS = 64
LR = 1e-3
PATIENCE = 4
CV_MAX_EPOCHS = 10       # the first build used 6, which every fold hit
FINAL_EPOCH_CAP = 8      # per-source fold best-epochs averaged 8.8; pooled CV lands lower
ENSEMBLE_SEEDS = (42, 43, 44)
N_FOLDS = 5
CV_TRAIN_CAP = 30000     # rows per CV fold, keeps the CV cheap

csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()
df["pos"] = np.arange(len(df))            # index into the pooled BIG/NTK arrays

rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    # np.array(g.index) copies: below pandas 3, to_numpy() shares the Index buffer and an
    # in-place shuffle would also reorder the labels the ranks below are read from.
    labels = np.array(g.index)
    rank = {lab: i for i, lab in enumerate(labels)}
    rng.shuffle(labels)
    n_test = max(1, int(round(0.2 * len(labels))))
    splits[proj] = {"test": np.array([rank[v] for v in labels[:n_test]]),
                    "train": np.array([rank[v] for v in labels[n_test:]])}

EMB = glob.glob("/kaggle/input/**/emb", recursive=True)[0]
META = EMB.replace("/emb", "/meta")


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


_files = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{EMB}/*.npy"))


def resolve(name):
    exact = safe(name)
    if exact in _files:
        return exact
    near = [f for f in _files if f.rstrip("_") == exact.rstrip("_")]
    assert len(near) == 1, f"no unique embedding file for {name!r}"
    return near[0]


names = sorted(df.project.unique())
N = len(df)
BIG = np.zeros((N, MAX_LEN, EMB_WIDTH), dtype=np.float16)
NTK = np.zeros(N, dtype=np.int32)
Y = df.storypoint.values.astype(np.float32)
YLOG = np.log1p(Y)
POS = {}
for proj in names:
    key = resolve(proj)
    emb = np.load(f"{EMB}/{key}.npy", mmap_mode="r")
    meta = pd.read_csv(f"{META}/{key}.csv")
    g = df[df.project == proj]
    assert emb.shape == (len(meta), MAX_LEN, EMB_WIDTH), (proj, emb.shape)
    assert len(meta) == len(g), f"{proj}: {len(meta)} meta rows vs {len(g)} filtered rows"
    # Row alignment is what makes positional indexing sound: same issue_ids, same order, same labels.
    assert (meta.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), \
        f"{proj}: embedding rows not aligned with all_issues_sp.csv"
    assert np.allclose(meta.storypoint.values, g.storypoint.values), f"{proj}: storypoint mismatch"
    gp = g.pos.values
    BIG[gp] = np.asarray(emb)
    NTK[gp] = np.clip(meta.n_tokens.values, 1, MAX_LEN)
    POS[proj] = gp
    del emb
print(f"aligned {len(names)} projects | pooled array {BIG.shape} ({BIG.nbytes / 1e9:.1f} GB)")

# Each target may train on every other project's development pool only.
pool_by_target = {t: np.concatenate([POS[p][splits[p]["train"]] for p in names if p != t])
                  for t in names}
pool_median_by_target = {t: float(np.median(Y[pool_by_target[t]])) for t in names}
train_all = np.concatenate([POS[p][splits[p]["train"]] for p in names])
GROUP = df.project.values                     # per-row project label, for grouped folds
print(f"pooled train rows: {len(train_all)} | mean pool per target: "
      f"{int(np.mean([len(v) for v in pool_by_target.values()]))}")

base = pd.read_csv(glob.glob("/kaggle/input/**/baseline_results.csv", recursive=True)[0])
base["source"] = base["source"].astype(str).str.strip()
base["target"] = base["target"].astype(str).str.strip()

# The pooled model and its per-target baselines must sit on the same partitions as 02. Compare each
# project's train-pool median with the constant 02 predicted from that pool.
within = base[base.target.str.contains("WITHIN")].set_index("source")
for proj in splits:
    pool = df[df.project == proj].storypoint.values[splits[proj]["train"]]
    med = float(np.median(pool))
    assert abs(med - float(within.loc[proj, "pred"])) < 1e-6, (proj, med, within.loc[proj, "pred"])
print("splits reproduce the baseline partitions for all 39 projects")

# %% [markdown]
# ## Model
#
# Identical to `05` — padding masked before the first convolution and after every layer, Softplus on
# the output. The only difference is the target: this model learns `log1p(story_point)` because the
# pooled label distribution is heavily skewed, and predictions are mapped back with `expm1`.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CNNRegressor(nn.Module):
    def __init__(self, d_in=EMB_WIDTH, ch=(256, 128, 64), p=0.2):
        super().__init__()
        c1, c2, c3 = ch
        self.conv1 = nn.Conv1d(d_in, c1, 3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv1d(c2, c3, 3, padding=1)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(c3, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, ntok):
        keep = torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]
        m = keep.unsqueeze(1)
        h = F.relu(self.conv1(x.transpose(1, 2) * m)) * m
        h = self.drop(h)
        h = F.relu(self.conv2(h)) * m
        h = F.relu(self.conv3(h)) * m
        pooled = h.masked_fill(~m, float("-inf")).max(dim=2).values
        return F.softplus(self.fc2(F.relu(self.fc1(pooled))).squeeze(-1))


def load(idx):
    x = torch.tensor(np.asarray(BIG[idx]), dtype=torch.float32, device=DEV)
    nt = torch.tensor(NTK[idx], dtype=torch.long, device=DEV)
    return x, nt


@torch.no_grad()
def predict_raw(model, idx, bs=512):
    """Story-point predictions (log1p target mapped back, clamped at 0)."""
    model.eval()
    out = []
    for b0 in range(0, len(idx), bs):
        x, nt = load(idx[b0:b0 + bs])
        out.append(torch.expm1(model(x, nt)).clamp(min=0).float().cpu().numpy())
    return np.concatenate(out)


def train_one(idx_pool, epochs, bs, seed, va_idx=None):
    """Train on global row indices `idx_pool`; with `va_idx` keep the best-validation-epoch weights
    and return (model, hist, best_val_mae_in_points, best_epoch_1_based)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNNRegressor().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.L1Loss()
    best_val, best_epoch, best_state = None, None, None
    bad_epochs, hist = 0, []
    for ep in range(epochs):
        model.train()
        perm = np.random.RandomState(seed * 1000 + ep).permutation(idx_pool)
        for b0 in range(0, len(perm), bs):
            sel = perm[b0:b0 + bs]
            x, nt = load(sel)
            y = torch.tensor(YLOG[sel], dtype=torch.float32, device=DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(x, nt), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if va_idx is None:
            continue
        vmae = float(np.abs(predict_raw(model, va_idx) - Y[va_idx]).mean())  # measured in points
        hist.append(vmae)
        if best_val is None or vmae < best_val:
            best_val, best_epoch, bad_epochs = vmae, ep + 1, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val, best_epoch

# %% [markdown]
# ## Project-grouped cross-validation
#
# Folds split by **project**, not by issue: a random fold would score a model on issues from projects
# it also trained on, which measures within-project fit rather than the cross-project transfer this
# project is about. Each fold also reports the project group it held out, so a target's epoch count
# can be averaged over only those folds that never held that target out.

# %%
from sklearn.model_selection import GroupKFold

holdout_rows = np.random.RandomState(SEED).choice(train_all, CV_TRAIN_CAP, replace=False)
gkf = GroupKFold(n_splits=N_FOLDS)
fold_records = []
oof_rows, oof_pred = [], []          # out-of-fold predictions, for the target-blind calibration
for f, (a, b) in enumerate(gkf.split(holdout_rows, groups=df.project.values[holdout_rows])):
    tr = holdout_rows[a]
    va = holdout_rows[b]
    held_out_projects = set(df.project.values[va].tolist())
    mdl, hist, vmae, vep = train_one(tr, epochs=CV_MAX_EPOCHS, bs=BS, seed=SEED + f, va_idx=va)
    fold_records.append(dict(fold=f, best_val_mae=vmae, best_epoch=vep, ran_epochs=len(hist),
                             held_out=held_out_projects))
    oof_rows.append(va)
    oof_pred.append(predict_raw(mdl, va))   # these rows were never trained on by this fold model
    print(f"fold {f + 1}: best val MAE {vmae:.3f} points at epoch {vep} of {len(hist)} run "
          f"(holds out {len(held_out_projects)} projects)", flush=True)
    del mdl
    if DEV == "cuda":
        torch.cuda.empty_cache()

oof_rows = np.concatenate(oof_rows)
oof_pred = np.concatenate(oof_pred)
oof_y = Y[oof_rows]
oof_proj = df.project.values[oof_rows]
cv_mae = np.array([r["best_val_mae"] for r in fold_records])
print(f"\ngrouped CV validation MAE: {cv_mae.mean():.3f} +/- {cv_mae.std():.3f} points "
      f"| best epochs {[r['best_epoch'] for r in fold_records]}")
print("a fold whose run count equals its best epoch never improved after that point; "
      "if every fold stops at CV_MAX_EPOCHS the ceiling, not the data, is choosing the epochs.")


def epochs_for_target(target):
    """Mean best epoch over the folds that did NOT hold this target out, so a target's
    configuration is chosen without its own development data."""
    use = [r["best_epoch"] for r in fold_records if target not in r["held_out"]]
    if not use:
        use = [r["best_epoch"] for r in fold_records]
    return int(np.clip(round(float(np.mean(use))), 1, FINAL_EPOCH_CAP))


# %% [markdown]
# ## Calibration (fitted on the other 38 projects' out-of-fold predictions only)
#
# Predictions trained on `log1p` labels and mapped back with `expm1` systematically underestimate on
# skewed targets, and the pooled model inherits the pool's scale. The correction is one affine map
# `a·p + b` per target, fitted by MAE on the grouped-CV **out-of-fold** predictions of the *other 38
# projects* — the target's own rows are excluded from the fit, so no target data enters training,
# epoch selection or calibration. For a fixed `a`, the MAE-optimal `b` is the median of the residuals
# `y − a·p`, so `a` is found by a grid search with that closed-form intercept.

# %%
def fit_calibration(p, y):
    """Return (a, b, mae_before) for p -> a*p + b, MAE-optimal on the given rows.
    For fixed a the MAE-optimal intercept is median(y - a*p)."""
    before = float(np.abs(p - y).mean())
    best_a, best_b, best_m = 1.0, 0.0, before
    for a in np.round(np.arange(0.20, 3.001, 0.02), 3):
        b = float(np.median(y - a * p))
        m = float(np.abs(a * p + b - y).mean())
        if m < best_m:
            best_a, best_b, best_m = float(a), b, m
    return best_a, best_b, before


def calibration_for_target(target):
    """Affine correction fitted on out-of-fold predictions of the OTHER projects only.
    The target's own rows (development and test) never enter the fit."""
    m = oof_proj != target
    if m.sum() < 100:
        return 1.0, 0.0, float("nan")
    return fit_calibration(oof_pred[m], oof_y[m])

# %% [markdown]
# ## Evaluate the 39 assigned pairs under the pooled protocol
#
# Each pair gets a 3-seed ensemble trained on the 38-project pool with its own target-blind epoch
# count and its own target-blind affine calibration (fitted on the other 38 projects' out-of-fold
# predictions). Two sets of metrics per pair: **uncalibrated** (strict zero-shot) and **calibrated**.
# Two baselines, both constants and both free of model training: the single-source median the
# assignment uses and the pooled median, which sees exactly the same 38 projects as the model.

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


assert len(PAIRS) == 39

os.makedirs("/kaggle/working/predictions_pooled", exist_ok=True)
base_idx = base.set_index(["source", "target"])
rows = []
t0 = time.time()
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    te_pos = POS[tgt][splits[tgt]["test"]]
    ep = epochs_for_target(tgt)
    a, b, oof_before = calibration_for_target(tgt)
    preds = []
    for sd in ENSEMBLE_SEEDS:
        mdl, _, _, _ = train_one(pool_by_target[tgt], epochs=ep, bs=BS, seed=sd)
        preds.append(predict_raw(mdl, te_pos))
        del mdl
        if DEV == "cuda":
            torch.cuda.empty_cache()
    pred = np.mean(preds, axis=0)
    pred_cal = np.maximum(a * pred + b, 0.0)

    yte = Y[te_pos]
    err = np.abs(pred - yte)
    err_cal = np.abs(pred_cal - yte)
    err_pool = np.abs(yte - pool_median_by_target[tgt])
    brow = base_idx.loc[(src, tgt)]
    rows.append(dict(pdf_source=s, pdf_target=t, source=src, target=tgt,
                     epochs=ep, n_test=int(len(yte)),
                     test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                     cal_mae=float(err_cal.mean()), cal_mdae=float(np.median(err_cal)),
                     cal_a=a, cal_b=b, oof_mae_before_cal=oof_before,
                     baseline_mae=float(brow.mae), baseline_mdae=float(brow.mdae),
                     pooled_median_mae=float(err_pool.mean()),
                     # the constants themselves, so per-issue baseline errors can be recomputed
                     # exactly (an MAE is not a prediction - see the Wilcoxon block below)
                     source_median_pred=float(brow.pred),
                     pooled_median_pred=float(pool_median_by_target[tgt]),
                     beats_source_baseline=bool(err.mean() < float(brow.mae)),
                     beats_pooled_baseline=bool(err.mean() < float(err_pool.mean())),
                     cal_beats_source_baseline=bool(err_cal.mean() < float(brow.mae)),
                     cal_beats_pooled_baseline=bool(err_cal.mean() < float(err_pool.mean()))))
    pd.DataFrame({
        "source_project": src, "target_project": tgt, "issue_id": df.issue_id.values[te_pos],
        "actual_story_point": yte, "predicted_story_point": pred,
        "predicted_calibrated": pred_cal, "absolute_error": err, "absolute_error_calibrated": err_cal,
        "source_median_prediction": float(brow.pred),
        "pooled_median_prediction": float(pool_median_by_target[tgt]),
    }).to_csv(f"/kaggle/working/predictions_pooled/{safe(tgt)}__seed_ens.csv", index=False)
    pd.DataFrame(rows).to_csv("/kaggle/working/results_pooled_pairs.csv", index=False)
    print(f"[{i + 1}/39] {s} -> {t}: MAE {rows[-1]['test_mae']:.3f} -> cal {rows[-1]['cal_mae']:.3f} "
          f"(a={a:.2f}, b={b:+.2f}; source median {rows[-1]['baseline_mae']:.3f}, pooled median "
          f"{rows[-1]['pooled_median_mae']:.3f}) "
          f"@{ep} ep | {time.time() - t0:.0f}s", flush=True)
print("ALL DONE in", round((time.time() - t0) / 60, 1), "min")

# %% [markdown]
# ## Summary
#
# `pooled_median_mae` is the reference both model rows are judged against: it is the same 38-project
# pool reduced to a single constant, with no target data and no training. The calibrated row applies
# the target-blind affine correction; the uncalibrated row shows what pooling alone buys. Beating the
# pooled median on the calibrated row is the claim that BERT text adds value over the pool's labels.

# %%
res = pd.DataFrame(rows)
preds_all = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("/kaggle/working/predictions_pooled/*.csv"))],
                      ignore_index=True)
macro = res.test_mae.mean()
macro_cal = res.cal_mae.mean()
weighted = float(np.average(res.test_mae, weights=res.n_test))
weighted_cal = float(np.average(res.cal_mae, weights=res.n_test))

# Paired significance: issue-level Wilcoxon on |error| of the calibrated model vs each constant
# baseline, and per-pair Wilcoxon across the 39 paired MAEs. scipy ships with Kaggle; skip the
# tests rather than the whole summary if it is missing.
try:
    from scipy.stats import wilcoxon
    # Per-issue baseline errors need the baseline PREDICTIONS (the constant medians), not their
    # MAEs. They are saved per row in the prediction files; res.pooled_median_pred /
    # res.source_median_pred are the fallback for prediction files written before this fix.
    tg = preds_all.target_project.values
    e_mod = preds_all.absolute_error_calibrated.values
    if "pooled_median_prediction" in preds_all.columns:
        e_pool = np.abs(preds_all.actual_story_point.values - preds_all.pooled_median_prediction.values)
        e_src = np.abs(preds_all.actual_story_point.values - preds_all.source_median_prediction.values)
    else:
        pm = dict(zip(res.target, res.pooled_median_pred))
        sm = dict(zip(res.target, res.source_median_pred))
        e_pool = np.abs(preds_all.actual_story_point.values - np.array([pm[t] for t in tg]))
        e_src = np.abs(preds_all.actual_story_point.values - np.array([sm[t] for t in tg]))
    p_pool = float(wilcoxon(e_mod, e_pool).pvalue)
    p_src = float(wilcoxon(e_mod, e_src).pvalue)
    p_pair_src = float(wilcoxon(res.cal_mae, res.baseline_mae).pvalue)
    p_pair_pool = float(wilcoxon(res.cal_mae, res.pooled_median_mae).pvalue)
    stats_lines = [
        f"issue-level Wilcoxon (calibrated vs pooled median): p = {p_pool:.2e} over {len(preds_all)} issues",
        f"issue-level Wilcoxon (calibrated vs source median): p = {p_src:.2e}",
        f"per-pair Wilcoxon over 39 pairs (cal MAE vs source median): p = {p_pair_src:.2e}",
        f"per-pair Wilcoxon over 39 pairs (cal MAE vs pooled median): p = {p_pair_pool:.2e}",
    ]
except Exception as ex:
    stats_lines = [f"(significance tests skipped: {type(ex).__name__}: {ex})"]

lines = [
    "Pooled-source BERT+CNN + target-blind calibration (multi-source extension, NOT the assigned single-source protocol)",
    f"seed {SEED} | log1p target | ensemble {ENSEMBLE_SEEDS} | grouped CV epochs, target-blind | "
    f"calibration on other-38 out-of-fold predictions only (no target data)",
    f"grouped CV val MAE {cv_mae.mean():.3f} +/- {cv_mae.std():.3f} | epochs used: "
    f"min {int(res.epochs.min())} mean {res.epochs.mean():.1f} max {int(res.epochs.max())}",
    "",
    f"macro test MAE over 39 pairs : uncalibrated {macro:.3f} +/- {res.test_mae.std():.3f} "
    f"| calibrated {macro_cal:.3f} +/- {res.cal_mae.std():.3f}",
    f"issue-weighted test MAE      : uncalibrated {weighted:.3f} | calibrated {weighted_cal:.3f}",
    f"pooled over {len(preds_all)} test issues: uncal MAE {preds_all.absolute_error.mean():.3f} "
    f"MdAE {preds_all.absolute_error.median():.3f} | cal MAE {preds_all.absolute_error_calibrated.mean():.3f} "
    f"MdAE {preds_all.absolute_error_calibrated.median():.3f}",
    "",
    f"UNCALIBRATED beats SINGLE-SOURCE median on {int(res.beats_source_baseline.sum())}/39 "
    f"(source-median macro {res.baseline_mae.mean():.3f})",
    f"UNCALIBRATED beats POOLED median on       {int(res.beats_pooled_baseline.sum())}/39 "
    f"(pooled-median macro {res.pooled_median_mae.mean():.3f})",
    f"CALIBRATED  beats SINGLE-SOURCE median on {int(res.cal_beats_source_baseline.sum())}/39",
    f"CALIBRATED  beats POOLED median on        {int(res.cal_beats_pooled_baseline.sum())}/39",
    "",
    *stats_lines,
    "",
    "ablation status: pooling, log1p target, the seed ensemble and calibration are combined in this",
    "run; separating them needs the A/B/C/D variants described in REPORT.md 'next steps'.",
]
txt = "\n".join(lines)
open("/kaggle/working/evaluation_summary_pooled.txt", "w").write(txt)
print(txt)

# %%
import matplotlib.pyplot as plt

plot = res.sort_values("cal_mae")
labels = [f"{s}  ->  {t}" for s, t in zip(plot.pdf_source, plot.pdf_target)]
ypos = np.arange(len(plot))
fig, ax = plt.subplots(figsize=(11, 13))
ax.barh(ypos + 0.26, plot.cal_mae, 0.26, color="tab:purple", label="pooled BERT + CNN, calibrated (3-seed)")
ax.barh(ypos, plot.pooled_median_mae, 0.26, color="tab:orange", alpha=0.75, label="pooled median (same 38 projects)")
ax.barh(ypos - 0.26, plot.baseline_mae, 0.26, color="lightgray", edgecolor="gray", label="single-source median baseline")
ax.set_yticks(ypos, labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("test MAE (story points)")
ax.set_title("Multi-source extension: calibrated pooled model vs pooled median vs single-source median\n"
             "different training data from the assigned PS-3 pairs — reported separately")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/results_pooled_per_pair.png", dpi=110)
plt.show()
