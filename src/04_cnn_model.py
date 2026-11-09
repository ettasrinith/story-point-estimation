# 04_cnn_model.py
#
# Walkthrough notebook: one cross-project pair end to end, with the training curves visible.
# Same model and same fold logic as 05 — 05 is the canonical run, this one is for reading the method.

# %% [markdown]
# # 04 — CNN Story Point Regressor on BERT Token Embeddings (demo pair)
#
# The three-layer CNN over frozen BERT token embeddings, trained and evaluated on a single
# cross-project pair (**Sonatype Nexus → Command-Line Interface**) so the whole method is visible in
# one place. `05_run_all_pairs` loops this exact code over all 39 assigned pairs.
#
# **What one pair does**
#
# 1. Train pool = the source project's 80% (seed-42 split, identical to the baseline in `02`).
# 2. 5-fold CV on that pool: train on 4 folds, validate on the 5th, early stopping, keep each fold's
#    **best** validation epoch.
# 3. Final model: train on the whole pool for the folds' mean best-epoch count.
# 4. Evaluate once on the target project's held-out 20%.
#
# **Inputs:** dataset `tawos-extracted` (`all_issues_sp.csv`, `baseline_results.csv`) and the output of
# kernel `tawos-bert-embeddings`. **Needs GPU** (a couple of minutes).
#
# This revision also fixes four things the earlier build got wrong: the fold metric came from the
# stopping epoch instead of the best epoch, padded tokens reached the convolutions, the head could
# output negative story points, and the split was not the seed-42 one — converting the shuffled draw
# into per-project ranks read the index buffer *after* the shuffle had reordered it, which made the
# test set the first 20% of the rows in file order instead of a random 20%. The split cell now checks
# the partitions against the baseline's own medians before anything trains.

# %%
import glob
import os
import re

import numpy as np
import pandas as pd

SEED = 42
MAX_LEN = 128
EMB_WIDTH = 768
BS = 64
LR = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4
N_FOLDS = 5
SRC, TGT = "Sonatype Nexus", "Command-Line Interface"

csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()

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


def load_project(name):
    """Memory-mapped token embeddings for one project, in the row order of its meta CSV."""
    key = safe(name)
    for cand in (key, key.rstrip("_")):
        if os.path.exists(f"{EMB}/{cand}.npy"):
            key = cand
            break
    emb = np.load(f"{EMB}/{key}.npy", mmap_mode="r")
    meta = pd.read_csv(f"{META}/{key}.csv")
    g = df[df.project == name]
    assert emb.shape == (len(meta), MAX_LEN, EMB_WIDTH), (name, emb.shape, len(meta))
    # The embeddings are indexed by row position, so the metadata must describe exactly the same
    # rows in the same order as the filtered issue frame. Verify, do not assume.
    assert len(meta) == len(g), f"{name}: {len(meta)} meta rows vs {len(g)} filtered rows"
    assert (meta.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), \
        f"{name}: embedding rows are not aligned with all_issues_sp.csv"
    return emb, meta, np.clip(meta.n_tokens.values, 1, MAX_LEN).astype(np.int64)


e, m, n = load_project(SRC)
print(f"{SRC}: embeddings {e.shape} | {len(m)} rows | test n = {len(splits[SRC]['test'])}")

base = pd.read_csv(glob.glob("/kaggle/input/**/baseline_results.csv", recursive=True)[0])
base["source"] = base["source"].astype(str).str.strip()
base["target"] = base["target"].astype(str).str.strip()
within = base[base.target.str.contains("WITHIN")].set_index("source")
for proj in (SRC, TGT):
    assert len(splits[proj]["test"]) == int(within.loc[proj, "n_test"]), proj
    # same sizes does not mean same rows: check which rows by comparing each train pool's median
    # with the constant the baseline predicted from that pool.
    pool = df[df.project == proj].storypoint.values[splits[proj]["train"]]
    med = float(np.median(pool))
    assert abs(med - float(within.loc[proj, "pred"])) < 1e-6, (proj, med, within.loc[proj, "pred"])
print("split sizes and source medians match the baseline for both projects")

# %% [markdown]
# ## Model and training loop
#
# `x` arrives as `[batch, 128, 768]` — one BERT vector per token position, right-padded to 128.
# `ntok` says how many of those positions are real.
#
# Padded positions carry a real (non-zero) BERT output, so they have to be handled explicitly:
#
# - zeroed **before** the first convolution — a kernel of width 3 reads neighbours, and without this
#   the last real tokens of a short issue would absorb padding activations;
# - re-zeroed after **each** convolution — otherwise the leak propagates through the layers;
# - excluded from max pooling, which otherwise collapses to whatever the padding produces.
#
# The head ends in Softplus: story points are positive by construction, and a plain linear output was
# free to predict negative ones.

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
        m_ = keep.unsqueeze(1)
        h = F.relu(self.conv1(x.transpose(1, 2) * m_)) * m_
        h = self.drop(h)
        h = F.relu(self.conv2(h)) * m_
        h = F.relu(self.conv3(h)) * m_
        pooled = h.masked_fill(~m_, float("-inf")).max(dim=2).values
        return F.softplus(self.fc2(F.relu(self.fc1(pooled))).squeeze(-1))


def to_device(X, y, ntok):
    return (torch.tensor(X, dtype=torch.float32, device=DEV),
            torch.tensor(y, dtype=torch.float32, device=DEV),
            torch.tensor(ntok, dtype=torch.long, device=DEV))


@torch.no_grad()
def model_predict(model, X, ntok, bs=512):
    model.eval()
    out = []
    for b0 in range(0, len(X), bs):
        x = torch.tensor(X[b0:b0 + bs], dtype=torch.float32, device=DEV)
        nt = torch.tensor(ntok[b0:b0 + bs], dtype=torch.long, device=DEV)
        out.append(model(x, nt).float().cpu().numpy())
    return np.concatenate(out)


def train_model(Xtr, ytr, ntr, epochs=MAX_EPOCHS, bs=BS, lr=LR, patience=PATIENCE, seed=0,
                Xva=None, yva=None, nva=None):
    """Returns (model, val_history, best_val_mae, best_epoch_1_based) — the model is left holding the
    weights of the best validation epoch whenever a validation set was given."""
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
        vmae = float(np.abs(model_predict(model, Xva, nva) - yva).mean())
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
# ## Run the demo pair

# %%
from sklearn.model_selection import KFold

Xs, ms, ns = load_project(SRC)
Xt, mt, nt_all = load_project(TGT)
Xtr = Xs[splits[SRC]["train"]]
ytr = ms.storypoint.values[splits[SRC]["train"]].astype(np.float32)
ntr = ns[splits[SRC]["train"]]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_vals, best_epochs, curves = [], [], []
for f, (a, b) in enumerate(kf.split(np.arange(len(ytr)))):
    mdl, hist, vmae, vep = train_model(Xtr[a], ytr[a], ntr[a], seed=SEED + f,
                                       Xva=Xtr[b], yva=ytr[b], nva=ntr[b])
    fold_vals.append(vmae)
    best_epochs.append(vep)
    curves.append(hist)
    print(f"fold {f + 1}: best val MAE {vmae:.3f} at epoch {vep} of {len(hist)} run")
    del mdl
    if DEV == "cuda":
        torch.cuda.empty_cache()

val_mean, val_std = float(np.mean(fold_vals)), float(np.std(fold_vals))
fix_epochs = int(np.clip(round(float(np.mean(best_epochs))), 1, MAX_EPOCHS))

mdl, _, _, _ = train_model(Xtr, ytr, ntr, epochs=fix_epochs, seed=SEED)
te_pos = splits[TGT]["test"]
pred = model_predict(mdl, Xt[te_pos], nt_all[te_pos])
ytest = mt.storypoint.values[te_pos].astype(np.float32)
err = np.abs(pred - ytest)
test_mae, test_mdae = float(err.mean()), float(np.median(err))

brow = base[(base.source == SRC) & (base.target == TGT)]
bmae = float(brow.mae.iloc[0]) if len(brow) else float("nan")

print(f"\n{SRC} -> {TGT}")
print(f"  {N_FOLDS}-fold best-epoch validation MAE: {val_mean:.3f} +/- {val_std:.3f}")
print(f"  fold best epochs {best_epochs} -> final model trained {fix_epochs} epochs on the full pool")
print(f"  test MAE {test_mae:.3f} (baseline {bmae:.3f}) | test MdAE {test_mdae:.3f} "
      f"| {len(ytest)} held-out issues")

pd.DataFrame([dict(source=SRC, target=TGT, val_mae_mean=val_mean, val_mae_std=val_std,
                   fold_best_epochs=";".join(map(str, best_epochs)), epochs=fix_epochs,
                   test_mae=test_mae, test_mdae=test_mdae, n_test=len(ytest), baseline_mae=bmae)]
             ).to_csv("/kaggle/working/demo_pair_result.csv", index=False)

# %% [markdown]
# ## Plots
#
# Left: the validation curve of every fold, with its best epoch marked — this is what the earlier
# build could not show, and it is why the reported fold number is the best epoch rather than the
# stopping epoch. Right: predictions against actuals on the held-out target.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for f, hist in enumerate(curves):
    ax[0].plot(range(1, len(hist) + 1), hist, marker="o", label=f"fold {f + 1}")
    ax[0].plot(best_epochs[f], fold_vals[f], "x", ms=11, mew=3, color=f"C{f}", alpha=0.9)
ax[0].axvline(fix_epochs, color="k", ls=":", label=f"final epochs = {fix_epochs}")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("validation MAE")
ax[0].set_title(f"{SRC}: 5-fold validation curves (x = best epoch)")
ax[0].legend(fontsize=7)

lims = [0, max(ytest.max(), pred.max()) + 1]
ax[1].scatter(ytest, pred, s=14, alpha=0.6)
ax[1].plot(lims, lims, "r--")
ax[1].set_xlabel("actual story point")
ax[1].set_ylabel("predicted")
ax[1].set_title(f"{SRC} -> {TGT}: test predictions")

ax[2].hist(err, bins=25)
ax[2].axvline(test_mae, color="r", ls="--", label=f"MAE {test_mae:.2f}")
ax[2].axvline(test_mdae, color="g", ls="--", label=f"MdAE {test_mdae:.2f}")
ax[2].set_xlabel("|predicted - actual|")
ax[2].set_title("absolute error on the target test set")
ax[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/demo_pair_plots.png", dpi=120)
plt.show()
