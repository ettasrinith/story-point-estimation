# 05_run_all_pairs.py
#
# Canonical implementation of the assigned PS-3 experiment: 39 source -> target pairs,
# per-source training, 5-fold CV on the source pool, held-out test on the target.
# 04_cnn_model.py runs one demo pair through the same code; 06 is a separate
# multi-source extension.
#
# %% [markdown]
# # 05 — Full Cross-Project Evaluation (39 assigned pairs)
#
# This is the primary PS-3 experiment: for every assigned pair, train the CNN regressor on the
# **source** project's development pool and test it once on the **target** project's held-out 20%.
#
# **Protocol per pair** (seed 42, the same 80:20 partitions as the median baseline in `02`, so every
# number below is directly comparable with `results/baseline_results.csv`):
#
# 1. 5-fold CV on the source train pool — train on 4 folds, validate on the 5th, early stopping.
# 2. Each fold keeps the weights of its **best** validation epoch; the reported fold MAE is that best
#    validation MAE and the fold contributes that **best epoch number** (not the stopping epoch).
# 3. Epoch count for the final model = mean of the five best epochs, then one model is trained on the
#    full source pool for that many epochs. The target project is never used for selection.
# 4. Single evaluation on the target's held-out test set.
#
# **What changed in this revision** (the previous version reported the last-epoch validation MAE and
# used the number of epochs actually run as the fold's "best epoch", so the epoch count came from the
# early-stopping point instead of the best model; padded tokens also reached the convolutions before
# masking, and the head could predict negative story points):
#
# - `train_model` returns `(model, hist, best_val_mae, best_epoch)` and the folds use those.
# - Padding is masked **before** the first convolution and re-zeroed after every layer.
# - The regression head ends in Softplus, so predictions are non-negative.
# - Embedding ↔ metadata ↔ source CSV rows are matched by `issue_id` for all 39 projects up front.
# - The split is checked against the baseline before training: a wrong partition costs nothing to
#   make and everything to notice later, so the cell compares each source's train-pool median with
#   the number the baseline predicted from that pool. (This caught a real one: on the Kaggle image's
#   pandas, `Index.to_numpy()` returns the index's own buffer, so shuffling the draw also reordered
#   the label array the per-project ranks were read from. The ranks then collapsed onto the shuffle
#   order and the "test set" became the first 20% of each project's rows in file order - a different
#   partition from the baseline's random 20%, which made the old comparison meaningless.)
# - Per-issue test predictions are saved, enabling a true pooled MdAE and issue-weighted MAE.
#
# **Inputs:** dataset `tawos-extracted` (`all_issues_sp.csv`, `baseline_results.csv`) and the output of
# kernel `tawos-bert-embeddings` (`emb/*.npy`, `meta/*.csv`). **Needs GPU** (T4, ~17 min for 39 pairs).
#
# **Outputs:** `results_pairs.csv`, `evaluation_summary.txt`, `results_per_pair.png`,
# `predictions/<source>__<target>.csv`.
#
# The 39 pairs are the project names from the assignment PDF. Two of them have no exact match in
# TAWOS and, with the instructor's approval, are mapped: "Evergreen" → "The MongoDB Engineering" and
# "C++ driver (Appcelerator)" → "Aptana Studio".

# %%
import glob
import os
import re
import time

import numpy as np
import pandas as pd

SEED = 42
MAX_LEN = 128          # must match the embedding extraction in 03
EMB_WIDTH = 768
BS = 64
LR = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4
N_FOLDS = 5

# Same filtering as 02 (baseline) and 03 (embeddings): numeric story points, keep (0, 40].
csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()   # three TAWOS names carry a trailing space

# Seed-42 80:20 split per project, rebuilt identically in 02/04/06. The splits are stored as
# row ranks *inside each project* because the embedding files are indexed per project.
# The label array must be a copy: below pandas 3, Index.to_numpy() hands out the Index's own
# buffer, so an in-place shuffle there rewrites g.index as well and the ranks taken from it
# afterwards point at the wrong rows. The partition check at the end of this cell catches it.
rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    labels = np.array(g.index)                  # copy: original row order, safe to shuffle
    rank = {lab: i for i, lab in enumerate(labels)}
    rng.shuffle(labels)
    n_test = max(1, int(round(0.2 * len(labels))))
    splits[proj] = {
        "test": np.array([rank[v] for v in labels[:n_test]]),
        "train": np.array([rank[v] for v in labels[n_test:]]),
    }

base = pd.read_csv(glob.glob("/kaggle/input/**/baseline_results.csv", recursive=True)[0])
base["source"] = base["source"].astype(str).str.strip()
base["target"] = base["target"].astype(str).str.strip()
within = base[base.target.str.contains("WITHIN")].set_index("source")
for proj in splits:
    assert len(splits[proj]["test"]) == int(within.loc[proj, "n_test"]), proj

# The baseline is only a fair comparator if this notebook and 02 drew the same partitions. Sizes
# matching is not enough proof - a different group order, or a different way of turning the draw
# into row positions, keeps the sizes and changes which issues are in which set. So check each
# source's train-pool median against the number the baseline actually predicted from that pool:
# equal medians over all 39 sources means the pools really are the same rows.
for proj in splits:
    train_pool = df[df.project == proj].storypoint.values[splits[proj]["train"]]
    med = float(np.median(train_pool))
    assert abs(med - float(within.loc[proj, "pred"])) < 1e-6, (proj, med, within.loc[proj, "pred"])
print(f"{len(df)} issues | {df.project.nunique()} projects | splits reproduce the baseline "
      "partitions (test sizes and source medians verified)")

# %% [markdown]
# ## Loading the frozen BERT representations
#
# `03` wrote one `[n_issues, 128, 768]` float16 array per project plus a `meta/*.csv` holding
# `issue_id, issue_key, storypoint, n_tokens` in the same row order. The arrays are memory-mapped, so
# the whole 12.2 GB never has to sit in RAM.
#
# Before any training, every project is checked three ways: the row count matches the filtered frame,
# the `issue_id` sequence matches it exactly, and the story points in the metadata equal the ones in
# the CSV. A mismatch would silently mislabel the whole experiment, so this fails fast instead of
# after 15 minutes of training.

# %%
EMB = glob.glob("/kaggle/input/**/emb", recursive=True)[0]
META = EMB.replace("/emb", "/meta")


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


_files = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{EMB}/*.npy"))
assert len(_files) == 39, f"expected 39 embedding files, found {len(_files)}"


def resolve(name):
    """Map a (stripped) project name onto its embedding file name."""
    exact = safe(name)
    if exact in _files:
        return exact
    # 03 saved files from the unstripped names, so trailing spaces became a trailing underscore.
    near = [f for f in _files if f.rstrip("_") == exact.rstrip("_")]
    assert len(near) == 1, f"no unique embedding file for {name!r} (looked for {exact!r}, got {near})"
    return near[0]


print("verifying embedding alignment for every project ...")
for proj in sorted(df.project.unique()):
    key = resolve(proj)
    meta = pd.read_csv(f"{META}/{key}.csv")
    g = df[df.project == proj]
    assert len(meta) == len(g), f"{proj}: {len(g)} filtered rows vs {len(meta)} meta rows"
    assert (meta.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), \
        f"{proj}: meta issue_id order does not match all_issues_sp.csv"
    assert np.allclose(meta.storypoint.values, g.storypoint.values), f"{proj}: storypoint mismatch"
print("all 39 projects aligned (issue_id + storypoint verified)")

_cache = {}


def load_project(name):
    if name not in _cache:
        key = resolve(name)
        emb = np.load(f"{EMB}/{key}.npy", mmap_mode="r")
        meta = pd.read_csv(f"{META}/{key}.csv")
        assert emb.shape == (len(meta), MAX_LEN, EMB_WIDTH), (name, emb.shape)
        # n_tokens is always >= 1 here (title is never empty), but the masked max-pool would return
        # -inf for a zero-token row, so it is clamped defensively.
        ntok = np.clip(meta.n_tokens.values, 1, MAX_LEN).astype(np.int64)
        _cache[name] = (emb, meta, ntok)
    return _cache[name]

# %% [markdown]
# ## Model
#
# Three 1D convolutions over the token axis (768→256→128→64, kernel 3, same-length padding), ReLU,
# dropout 0.2 after the first block, masked global max pooling, then a 64→32→1 MLP head closed by
# Softplus.
#
# The mask matters twice over. Convolutions read neighbouring positions, so padded positions have to
# be zeroed *before* the first layer or their activations smear into the last real tokens; and max
# pooling has to ignore them afterwards or the padding's constant activation wins the pool. Zeroing
# after every layer keeps both clean, which the previous build only did once, after all three
# convolutions.

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

    def forward(self, x, ntok):                     # x [B, L, 768], ntok [B]
        keep = torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]
        m = keep.unsqueeze(1)                       # [B, 1, L]
        h = self.conv1(x.transpose(1, 2) * m)       # padded embeddings never enter the convolutions
        h = F.relu(h) * m
        h = self.drop(h)
        h = F.relu(self.conv2(h)) * m
        h = F.relu(self.conv3(h)) * m
        h = h.masked_fill(~m, float("-inf"))        # pool over real tokens only
        pooled = h.max(dim=2).values
        return F.softplus(self.fc2(F.relu(self.fc1(pooled))).squeeze(-1))


def to_device(X, y, ntok):
    return (torch.tensor(X, dtype=torch.float32, device=DEV),
            torch.tensor(y, dtype=torch.float32, device=DEV),
            torch.tensor(ntok, dtype=torch.long, device=DEV))


@torch.no_grad()
def predict(model, X, ntok, bs=512):
    model.eval()
    out = []
    for b0 in range(0, len(X), bs):
        x = torch.tensor(X[b0:b0 + bs], dtype=torch.float32, device=DEV)
        nt = torch.tensor(ntok[b0:b0 + bs], dtype=torch.long, device=DEV)
        out.append(model(x, nt).float().cpu().numpy())
    return np.concatenate(out)


def train_model(Xtr, ytr, ntr, epochs=MAX_EPOCHS, bs=BS, lr=LR, patience=PATIENCE, seed=0,
                Xva=None, yva=None, nva=None):
    """Train one model. With a validation set, the best-validation-epoch weights are restored and
    (model, val_history, best_val_mae, best_epoch) is returned; without one, (model, [], None, None)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNNRegressor().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()

    best_val, best_epoch, best_state = None, None, None
    bad_epochs, hist = 0, []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(ytr))
        for b0 in range(0, len(ytr), bs):
            sel = perm[b0:b0 + bs]
            x, y, nt = to_device(Xtr[sel], ytr[sel], ntr[sel])
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(x, nt), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if Xva is None:
            continue
        vmae = float(np.abs(predict(model, Xva, nva) - yva).mean())
        hist.append(vmae)
        if best_val is None or vmae < best_val:
            best_val, best_epoch, bad_epochs = vmae, ep + 1, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val, best_epoch

# %% [markdown]
# ## One pair end to end
#
# `run_pair` returns the fold-level validation numbers, the epoch count chosen from the folds, the
# held-out test MAE / MdAE and the target's test-set size (needed later for the issue-weighted
# average). It also writes the issue-level predictions for that pair.

# %%
from sklearn.model_selection import KFold

PRED_DIR = "/kaggle/working/predictions"
os.makedirs(PRED_DIR, exist_ok=True)


def run_pair(src, tgt, seed=SEED):
    Xs, ms, ns = load_project(src)
    Xt, mt, nt_all = load_project(tgt)
    tr_pos, te_pos = splits[src]["train"], splits[tgt]["test"]

    Xtr, ytr, ntr = Xs[tr_pos], ms.storypoint.values[tr_pos].astype(np.float32), ns[tr_pos]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_vals, best_epochs = [], []
    for f, (a, b) in enumerate(kf.split(np.arange(len(ytr)))):
        mdl, hist, vmae, vep = train_model(
            Xtr[a], ytr[a], ntr[a], seed=seed + f,
            Xva=Xtr[b], yva=ytr[b], nva=ntr[b])
        fold_vals.append(vmae)
        best_epochs.append(vep)
        print(f"    fold {f + 1}: best val MAE {vmae:.3f} at epoch {vep} "
              f"({len(hist)} epochs run)", flush=True)
        del mdl, hist
        if DEV == "cuda":
            torch.cuda.empty_cache()

    # best_epoch is reported 1-based, so its mean is already the epoch count to train for.
    fix_epochs = int(np.clip(round(float(np.mean(best_epochs))), 1, MAX_EPOCHS))
    mdl, _, _, _ = train_model(Xtr, ytr, ntr, epochs=fix_epochs, seed=seed)

    Xte, yte, nte = Xt[te_pos], mt.storypoint.values[te_pos].astype(np.float32), nt_all[te_pos]
    pred = predict(mdl, Xte, nte)
    err = np.abs(pred - yte)

    pd.DataFrame({
        "source_project": src, "target_project": tgt,
        "issue_id": mt.issue_id.values[te_pos], "issue_key": mt.issue_key.values[te_pos],
        "actual_story_point": yte, "predicted_story_point": pred, "absolute_error": err,
    }).to_csv(f"{PRED_DIR}/{safe(src)}__{safe(tgt)}.csv", index=False)

    return {
        "val_mae_mean": float(np.mean(fold_vals)),
        "val_mae_std": float(np.std(fold_vals)),
        "val_best_epoch_mean": float(np.mean(best_epochs)),
        "epochs": fix_epochs,
        "test_mae": float(err.mean()),
        "test_mdae": float(np.median(err)),
        "n_test": int(len(yte)),
    }

# %% [markdown]
# ## The 39 assigned pairs
#
# `PAIRS` is the assignment table in its given order (PDF names); `NAME` maps a PDF label onto the
# TAWOS project name when they differ.

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

# %% [markdown]
# ## Run all 39 pairs
#
# Results are rewritten after every pair, so a timeout or an interrupted session still leaves a
# complete CSV of everything finished so far.

# %%
base_idx = base.set_index(["source", "target"])
rows = []
t0 = time.time()
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    try:
        r = run_pair(src, tgt)
        status = "ok"
    except Exception as ex:
        print(f"[{i + 1}/39] {s} -> {t}: FAILED - {type(ex).__name__}: {ex}", flush=True)
        r = {k: float("nan") for k in
             ("val_mae_mean", "val_mae_std", "val_best_epoch_mean", "test_mae", "test_mdae")}
        r["epochs"] = 0
        r["n_test"] = 0
        status = f"failed:{type(ex).__name__}"
    brow = base_idx.loc[(src, tgt)]
    r.update(pdf_source=s, pdf_target=t, source=src, target=tgt, status=status,
             baseline_mae=float(brow.mae), baseline_mdae=float(brow.mdae))
    rows.append(r)
    pd.DataFrame(rows).to_csv("/kaggle/working/results_pairs.csv", index=False)
    if status == "ok":
        print(f"[{i + 1}/39] {s} -> {t}: test MAE {r['test_mae']:.3f} "
              f"(baseline {r['baseline_mae']:.3f}), MdAE {r['test_mdae']:.3f}, "
              f"val {r['val_mae_mean']:.3f}+/-{r['val_mae_std']:.3f} @ep{r['val_best_epoch_mean']:.1f} "
              f"-> {r['epochs']} epochs | {time.time() - t0:.0f}s", flush=True)
print("ALL PAIRS DONE in", round((time.time() - t0) / 60, 1), "min")

# %% [markdown]
# ## Summary
#
# Two averages are reported because they answer different questions. **Macro** MAE gives each of the
# 39 pairs equal weight; **issue-weighted** MAE weights a pair by the size of its test set (they run
# from 40 to 3,872 issues). The pooled MdAE is the median of all 12k individual absolute errors,
# which is not the same as averaging the 39 per-pair medians.

# %%
res = pd.DataFrame(rows)
ok = res[res.status == "ok"].copy()
preds = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{PRED_DIR}/*.csv"))], ignore_index=True)

macro_mae = ok.test_mae.mean()
macro_std = ok.test_mae.std()
weighted_mae = float(np.average(ok.test_mae, weights=ok.n_test))
pooled_mdae = float(preds.absolute_error.median())
pooled_mae = float(preds.absolute_error.mean())
lines = [
    "PS-3 cross-project evaluation - BERT word embeddings + 3-layer CNN (corrected protocol)",
    f"seed {SEED} | SP in (0,40] | same 80:20 partitions as baseline | {N_FOLDS}-fold CV on source pool",
    "fold metric = best validation epoch; final epochs = mean of the five best epochs",
    "",
    f"pairs evaluated: {len(ok)} of {len(res)} assigned",
    f"macro (pair-average) test MAE : {macro_mae:.3f} +/- {macro_std:.3f}   (baseline {ok.baseline_mae.mean():.3f})",
    f"issue-weighted test MAE       : {weighted_mae:.3f}   (baseline {float(np.average(ok.baseline_mae, weights=ok.n_test)):.3f})",
    f"pooled over all {len(preds)} test issues: MAE {pooled_mae:.3f} | MdAE {pooled_mdae:.3f}",
    f"mean of per-pair test MdAE    : {ok.test_mdae.mean():.3f}   (baseline {ok.baseline_mdae.mean():.3f})",
    f"best-epoch validation MAE     : {ok.val_mae_mean.mean():.3f} +/- {ok.val_mae_std.mean():.3f}",
    f"epochs used for final model   : mean {ok.epochs.mean():.1f} (min {int(ok.epochs.min())}, max {int(ok.epochs.max())})",
    f"beats baseline MAE on         : {int((ok.test_mae < ok.baseline_mae).sum())}/{len(ok)} pairs",
    f"pairs where the baseline median is unbeatable (MAE 0): {int((ok.baseline_mae == 0).sum())}",
    "",
    "best 5 pairs by test MAE:",
    ok.nsmallest(5, "test_mae")[["pdf_source", "pdf_target", "test_mae", "test_mdae",
                                  "baseline_mae", "n_test"]].to_string(index=False),
    "",
    "worst 5 pairs by test MAE:",
    ok.nlargest(5, "test_mae")[["pdf_source", "pdf_target", "test_mae", "test_mdae",
                                 "baseline_mae", "n_test"]].to_string(index=False),
]
txt = "\n".join(lines)
open("/kaggle/working/evaluation_summary.txt", "w").write(txt)
print(txt)

# %% [markdown]
# ## Plot
#
# Sorted horizontal bars with full project names, so a reader can see which transfers succeeded —
# the earlier vertical bars truncated every label to nine characters.

# %%
import matplotlib.pyplot as plt

plot = ok.sort_values("test_mae", ascending=True)
labels = [f"{s}  ->  {t}" for s, t in zip(plot.pdf_source, plot.pdf_target)]
fig, ax = plt.subplots(figsize=(11, 13))
ypos = np.arange(len(plot))
ax.barh(ypos + 0.2, plot.test_mae, 0.4, color="tab:blue", label="BERT + CNN (per source)")
ax.barh(ypos - 0.2, plot.baseline_mae, 0.4, color="lightgray", edgecolor="gray", label="median baseline")
ax.set_yticks(ypos, labels, fontsize=8)
ax.axvline(macro_mae, color="tab:blue", ls=":", lw=1, label=f"model macro MAE {macro_mae:.2f}")
ax.axvline(ok.baseline_mae.mean(), color="gray", ls="--", lw=1,
           label=f"baseline macro MAE {ok.baseline_mae.mean():.2f}")
ax.invert_yaxis()
ax.set_xlabel("test MAE (story points)")
ax.set_title("Assigned PS-3 pairs: per-source BERT + CNN vs median baseline\n"
             "39 source -> target transfers, seed 42, corrected 5-fold protocol")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/results_per_pair.png", dpi=110)
plt.show()
