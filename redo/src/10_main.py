# 10_main.py
#
# PS-3 redo, main assigned experiment. Configuration:
#   BERT token window 256 (was 128), story points in (0, 100] (was (0, 40]).
# Everything else matches the corrected protocol of 05: seed 42, 80:20 random split,
# 5-fold CV on the source development pool, best-validation-epoch selection, one held-out
# evaluation on the target's 20%.

# %% [markdown]
# # 10 — PS-3 assigned experiment at 256 tokens / story points (0,100]
#
# This source builds **two** Kaggle kernels from the same code:
#
# | kernel | window | output tag | purpose |
# |---|---|---|---|
# | `ps3-256-main` | full 256 tokens | `256` | the requested redo configuration |
# | `ps3-256-len128` | first 128 positions of the same arrays | `128view` | controlled before/after ablation |
#
# The second one changes **only** the token window: identical corpus, splits, seeds, BERT
# weights, CNN and protocol. That makes the 128 → 256 comparison a measurement rather than a
# comparison of two different published result sets.
#
# The kernel is self-contained: rather than trusting the old `baseline_results.csv` (computed
# for the (0,40] filter), it recomputes the median baseline, the seed-42 partitions, the data
# audit and the truncation statistics from the same corpus the 256-token arrays came from.
#
# **Inputs**
# - dataset `ettasrinithreddy/tawos-extracted` → `all_issues_sp.csv`
# - kernel outputs `madmaxop123/tawos-bert-256-a` and `madmaxop123/tawos-bert-256-b`
#   → `emb256/<PROJECT>.npy` float16 `[n, 256, 768]` + `meta/<PROJECT>.csv`
#
# **Configuration** (all fixed, recorded in `run_metadata_<tag>.json`): seed 42, 80:20 split,
# 5 folds, `bert-base-uncased` frozen token embeddings, story points in (0, 100], three-layer
# 1-D CNN, L1 loss, Adam 1e-3, batch 64, max 25 epochs, patience 4, Softplus head, fp16
# autocast on a T4.
#
# **Outputs** (tagged by window so they can never be confused with the committed 128/(0,40]
# run): data audit, baseline, per-pair metrics, per-issue predictions, summary and plots.
#
# **GPU + no Internet required** (the embeddings are already stored).

# %%
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd

SEED = 42
SP_MIN, SP_MAX = 0, 100
# The builder injects CONFIG_OVERRIDE for the controlled 128-token comparison kernel.
# Both runs execute exactly the same code below; only the window and the output tag differ.
try:
    CONFIG_OVERRIDE
except NameError:
    CONFIG_OVERRIDE = {}
STORED_LEN = 256
MAX_LEN = int(CONFIG_OVERRIDE.get("max_len", STORED_LEN))
TAG = CONFIG_OVERRIDE.get("tag", "256")
SUF = f"_{TAG}"
EMB_WIDTH = 768
BS = 64
LR = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4
N_FOLDS = 5
OUT = "/kaggle/working"
PRED_DIR = f"{OUT}/predictions{SUF}"
os.makedirs(PRED_DIR, exist_ok=True)

t_start = time.time()
print("config:", dict(tag=TAG, seed=SEED, sp_filter=f"({SP_MIN},{SP_MAX}]", max_len=MAX_LEN,
                      folds=N_FOLDS, batch=BS, lr=LR, max_epochs=MAX_EPOCHS, patience=PATIENCE))

# %% [markdown]
# ## 1. Corpus and the (0,100] label filter
#
# The old run kept story points in (0,40]. The redo uses the standard Deep-SE cap (0,100]:
# zero-point tickets and values above 100 are dropped. Both counts are recorded so the
# change in the corpus is auditable rather than hidden inside a filter expression.

# %%
csv_hits = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)
assert csv_hits, "attach the tawos-extracted dataset"
raw = pd.read_csv(csv_hits[0])
raw["storypoint"] = pd.to_numeric(raw["storypoint"], errors="coerce")
n_with_sp = int(raw.storypoint.notna().sum())
sp = raw.storypoint.dropna()

audit = dict(
    rows_in_csv=int(len(raw)),
    rows_with_storypoint=n_with_sp,
    rows_without_storypoint=int(len(raw) - n_with_sp),
    dropped_sp_le_0=int((sp <= 0).sum()),
    dropped_sp_gt_100=int((sp > 100).sum()),
    dropped_sp_gt_40=int((sp > 40).sum()),
    kept_sp_0_40=int(((sp > 0) & (sp <= 40)).sum()),
    kept_sp_0_100=int(((sp > 0) & (sp <= 100)).sum()),
    max_storypoint=float(sp.max()),
    added_by_relaxing_40_to_100=int(((sp > 40) & (sp <= 100)).sum()),
)
print(json.dumps(audit, indent=2))

df = raw.dropna(subset=["storypoint"]).copy()
df = df[(df.storypoint > SP_MIN) & (df.storypoint <= SP_MAX)].copy()
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()
print("filtered corpus:", len(df), "issues |", df.project.nunique(), "projects")
assert df.project.nunique() == 39, df.project.nunique()

# Per-project manifest and the cost of both filters (40 and 100), for the comparative report.
spdf = raw.dropna(subset=["storypoint"]).copy()
spdf["project"] = spdf["project"].astype(str).str.strip()
rows = []
for proj, g in spdf.groupby("project"):
    s = g.storypoint
    rows.append(dict(project=proj, raw=len(g),
                     kept_0_40=int(((s > 0) & (s <= 40)).sum()),
                     kept_0_100=int(((s > 0) & (s <= 100)).sum()),
                     added=int(((s > 40) & (s <= 100)).sum()),
                     dropped_zero=int((s <= 0).sum()),
                     dropped_gt_100=int((s > 100).sum()),
                     median_sp=float(s[(s > 0) & (s <= SP_MAX)].median())))
manifest = pd.DataFrame(rows).sort_values("project")
manifest.to_csv(f"{OUT}/manifest{SUF}.csv", index=False)
json.dump(audit, open(f"{OUT}/filter_impact{SUF}.json", "w"), indent=2)

# %% [markdown]
# ## 2. Seed-42 80:20 partitions and the median baseline, recomputed
#
# The same construction as 02/05 (a single `default_rng(42)` walked over projects in sorted
# group order), but on the new filtered corpus. Splits are stored as row positions *inside each
# project*, because the embedding arrays are per project. The training notebooks never see the
# target's rows until the final evaluation.

# %%
rng = np.random.default_rng(SEED)
splits, frames = {}, {}
for proj in sorted(df.project.unique()):
    g = df[df.project == proj].reset_index(drop=True)
    frames[proj] = g
    labels = np.array(g.index)
    rank = {lab: i for i, lab in enumerate(labels)}
    rng.shuffle(labels)
    n_test = max(1, int(round(0.2 * len(labels))))
    splits[proj] = {
        "test": np.array([rank[v] for v in labels[:n_test]]),
        "train": np.array([rank[v] for v in labels[n_test:]]),
    }
projects = sorted(splits)
medians = {p: float(np.median(frames[p].storypoint.values[splits[p]["train"]])) for p in projects}

base_rows = []
for src in projects:
    for tgt in projects:
        if tgt == src:
            continue
        y = frames[tgt].storypoint.values[splits[tgt]["test"]]
        err = np.abs(y - medians[src])
        base_rows.append(dict(source=src, target=tgt, mae=float(err.mean()),
                              mdae=float(np.median(err)), pred=medians[src],
                              n_train=len(splits[src]["train"]), n_test=len(y)))
for p in projects:
    y = frames[p].storypoint.values[splits[p]["test"]]
    err = np.abs(y - medians[p])
    base_rows.append(dict(source=p, target=f"{p} [WITHIN]", mae=float(err.mean()),
                          mdae=float(np.median(err)), pred=medians[p],
                          n_train=len(splits[p]["train"]), n_test=len(y)))
base = pd.DataFrame(base_rows)
base.to_csv(f"{OUT}/baseline_results{SUF}.csv", index=False)
base_idx = base.set_index(["source", "target"])
cross, within = base[~base.target.str.contains("WITHIN")], base[base.target.str.contains("WITHIN")]

# Like-for-like constant for the pooled extension: median over the *other 38* projects' pools.
pooled_median = {}
for tgt in projects:
    vals = np.concatenate([frames[p].storypoint.values[splits[p]["train"]]
                           for p in projects if p != tgt])
    pooled_median[tgt] = float(np.median(vals))

btxt = "\n".join([
    f"seed {SEED} | story points ({SP_MIN},{SP_MAX}] | {len(df)} issues | {len(projects)} projects",
    f"CROSS-PROJECT  (n={len(cross)}): MAE {cross.mae.mean():.3f} +/- {cross.mae.std():.3f} | MdAE {cross.mdae.mean():.3f}",
    f"WITHIN-PROJECT (n={len(within)}): MAE {within.mae.mean():.3f} +/- {within.mae.std():.3f} | MdAE {within.mdae.mean():.3f}",
    f"cross/within gap: {100 * (cross.mae.mean() / within.mae.mean() - 1):.1f}%",
])
open(f"{OUT}/baseline_summary{SUF}.txt", "w").write(btxt)
print(btxt)

# %% [markdown]
# ## 3. Locate and verify the 256-token BERT embeddings
#
# The extraction was split across two kernels because 24 GB does not fit one kernel output.
# This cell discovers the mount paths rather than hard-coding them, then checks every project:
# shape `[n, 256, 768]`, row count, `issue_id` order and story-point values against the filtered
# corpus. Any mismatch aborts the run before a single model is trained.

# %%
all_npy = sorted(glob.glob("/kaggle/input/**/*.npy", recursive=True))
all_csv = sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
print("found", len(all_npy), "npy files in", sorted({p.split("/")[3] for p in all_npy}))
assert len(all_npy) == 39, f"expected 39 embedding arrays, found {len(all_npy)}"

npy_by_key = {os.path.basename(p)[:-4]: p for p in all_npy}
csv_by_key = {os.path.basename(p)[:-4]: p for p in all_csv}
assert len(npy_by_key) == 39, "duplicate embedding file names across the two halves"


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def resolve(name):
    exact = safe(name)
    if exact in npy_by_key:
        return exact
    near = [k for k in npy_by_key if k.rstrip("_") == exact.rstrip("_")]
    assert len(near) == 1, f"no unique embedding for {name!r}: {near}"
    return near[0]


print("verifying alignment for all 39 projects ...")
for proj in projects:
    key = resolve(proj)
    assert key in csv_by_key, f"missing metadata for {proj}"
    m = pd.read_csv(csv_by_key[key])
    g = frames[proj]
    assert len(m) == len(g), (proj, len(m), len(g))
    assert (m.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), proj
    assert np.allclose(m.storypoint.values, g.storypoint.values), proj
    a = np.load(npy_by_key[key], mmap_mode="r")
    assert a.shape == (len(m), STORED_LEN, EMB_WIDTH), (proj, a.shape)
    assert np.isfinite(np.asarray(a[:2])).all(), proj
print(f"all 39 arrays aligned: [n, {STORED_LEN}, {EMB_WIDTH}], float16, finite, "
      "issue_id + storypoint verified")


class WindowView:
    """Lazy `X[rows]` that returns only the first `width` token positions.

    Used by the controlled 128-token kernel: the very same BERT weights, corpus, splits and
    training code, with the sequence truncated to the old window. For an issue longer than
    127 tokens the 256-token encoding has its [SEP] at the end, so this view omits it; that
    is documented as the one unavoidable difference from a fresh 128-token extraction.
    """

    def __init__(self, arr, width):
        self.arr, self.width = arr, width

    def __getitem__(self, rows):
        return np.asarray(self.arr[rows])[:, :self.width, :]


_cache = {}


def load_project(name):
    if name not in _cache:
        key = resolve(name)
        emb = np.load(npy_by_key[key], mmap_mode="r")
        X = emb if MAX_LEN == STORED_LEN else WindowView(emb, MAX_LEN)
        meta = pd.read_csv(csv_by_key[key])
        y = meta.storypoint.values.astype(np.float32)
        ntok = np.clip(meta.n_tokens.values, 1, MAX_LEN).astype(np.int64)
        _cache[name] = (X, meta, y, ntok)
    return _cache[name]

# %% [markdown]
# ## 4. Truncation audit: what the 256-token window actually buys
#
# `n_tokens` in the metadata is the **untruncated** tokenizer length, so the same column
# measures both windows on the same corpus. This is the honest, measurable motivation for
# the redo (no estimate, no extrapolation).

# %%
tr_rows, lens_all = [], []
for proj in projects:
    emb, meta, y, ntok = load_project(proj)
    L = meta.n_tokens.values.astype(np.int64)
    lens_all.append(L)
    tr_rows.append(dict(
        project=proj, n=len(L),
        median_tokens=float(np.median(L)), p90_tokens=float(np.percentile(L, 90)),
        trunc_pct_128=round(float((L > 128).mean() * 100), 2),
        trunc_pct_256=round(float((L > 256).mean() * 100), 2),
        content_lost_at_128_pct=round(float(np.maximum(L - 128, 0).sum() / L.sum() * 100), 2),
        content_lost_at_256_pct=round(float(np.maximum(L - 256, 0).sum() / L.sum() * 100), 2),
    ))
tr = pd.DataFrame(tr_rows).sort_values("trunc_pct_128", ascending=False)
tr.to_csv(f"{OUT}/truncation_by_project{SUF}.csv", index=False)
L = np.concatenate(lens_all)
tstats = dict(
    issues=int(len(L)), median_tokens=float(np.median(L)),
    p90_tokens=float(np.percentile(L, 90)), p99_tokens=float(np.percentile(L, 99)),
    trunc_pct_128=round(float((L > 128).mean() * 100), 2),
    trunc_pct_256=round(float((L > 256).mean() * 100), 2),
    content_lost_at_128_pct=round(float(np.maximum(L - 128, 0).sum() / L.sum() * 100), 2),
    content_lost_at_256_pct=round(float(np.maximum(L - 256, 0).sum() / L.sum() * 100), 2),
)
json.dump(tstats, open(f"{OUT}/truncation_stats{SUF}.json", "w"), indent=2)
print(json.dumps(tstats, indent=2))
print(tr.head(6).to_string(index=False))

# %% [markdown]
# ## 5. Model — three-layer CNN over masked token embeddings
#
# Identical architecture to 05/06/07 (768→256→128→64 channels, kernel 3, ReLU, dropout 0.2,
# masked global max pool, 64→32→1 head ending in Softplus). Padding is zeroed **before** the
# first convolution and after every layer, so padded positions cannot leak into real tokens or
# win the max pool. The only change in this kernel is that rows are drawn lazily from the
# memory-mapped arrays by row position — at 256 tokens a materialised copy of a large project's
# training pool would be several GB.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("device:", DEV, "|", torch.cuda.get_device_name(0) if DEV == "cuda" else "CPU")


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
        h = self.conv1(x.transpose(1, 2) * m)
        h = F.relu(h) * m
        h = self.drop(h)
        h = F.relu(self.conv2(h)) * m
        h = F.relu(self.conv3(h)) * m
        pooled = h.masked_fill(~m, float("-inf")).max(dim=2).values
        return F.softplus(self.fc2(F.relu(self.fc1(pooled))).squeeze(-1))


def batch(X, rows, y, ntok):
    return (torch.tensor(np.asarray(X[rows]), dtype=torch.float32, device=DEV),
            torch.tensor(y[rows], dtype=torch.float32, device=DEV),
            torch.tensor(ntok[rows], dtype=torch.long, device=DEV))


@torch.no_grad()
def predict(model, X, rows, ntok, bs=256):
    model.eval()
    out = []
    for b0 in range(0, len(rows), bs):
        sel = rows[b0:b0 + bs]
        x = torch.tensor(np.asarray(X[sel]), dtype=torch.float32, device=DEV)
        nt = torch.tensor(ntok[sel], dtype=torch.long, device=DEV)
        out.append(model(x, nt).float().cpu().numpy())
    return np.concatenate(out)


def train_model(X, rows, y, ntok, epochs=MAX_EPOCHS, rows_va=None, seed=0,
                bs=BS, lr=LR, patience=PATIENCE):
    """Fit one CNN. With a validation set, restore the best-validation-epoch weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CNNRegressor().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    best_val = best_epoch = best_state = None
    bad, hist = 0, []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(rows))
        for b0 in range(0, len(rows), bs):
            sel = rows[perm[b0:b0 + bs]]
            x, yy, nt = batch(X, sel, y, ntok)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(x, nt), yy)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if rows_va is None:
            continue
        vmae = float(np.abs(predict(model, X, rows_va, ntok) - y[rows_va]).mean())
        hist.append(vmae)
        if best_val is None or vmae < best_val:
            best_val, best_epoch, bad = vmae, ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_val, best_epoch

# %% [markdown]
# ## 6. The 39 assigned source → target pairs
#
# The pair list is the assignment table verbatim. Two labels have no exact match in TAWOS and
# use the instructor-approved mappings from the original report: Evergreen →
# The MongoDB Engineering, C++ driver (Appcelerator) → Aptana Studio.

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
assert len(PAIRS) == 39 and len(set(PAIRS)) == 39
_pairs_in_baseline = set(base_idx.index.tolist())
for s, t in PAIRS:
    assert (dsn(s), dsn(t)) in _pairs_in_baseline, (s, t)

# %% [markdown]
# ## 7. Run every pair
#
# Results and per-issue predictions are written after each pair, so an interrupted run still
# leaves a complete record of everything it finished. Validation (5-fold, best epoch) is
# always reported separately from the single held-out test evaluation, per §3.4.

# %%
def run_pair(src, tgt, seed=SEED):
    Xs, ms, ys, ns = load_project(src)
    Xt, mt, yt, nt = load_project(tgt)
    tr, te = splits[src]["train"], splits[tgt]["test"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_vals, best_epochs, fold_stds = [], [], []
    for f, (a, b) in enumerate(kf.split(tr)):
        mdl, hist, vmae, vep = train_model(Xs, tr[a], ys, ns, rows_va=tr[b], seed=seed + f)
        fold_vals.append(vmae)
        best_epochs.append(vep)
        fold_stds.append(float(np.std(hist)) if hist else float("nan"))
        print(f"    fold {f+1}: best val MAE {vmae:.3f} @ epoch {vep} ({len(hist)} run)", flush=True)
        del mdl
        if DEV == "cuda":
            torch.cuda.empty_cache()

    fix_epochs = int(np.clip(round(float(np.mean(best_epochs))), 1, MAX_EPOCHS))
    mdl, _, _, _ = train_model(Xs, tr, ys, ns, epochs=fix_epochs, seed=seed)
    pred = predict(mdl, Xt, te, nt)
    actual = yt[te]
    err = np.abs(pred - actual)
    bconst = medians[src]
    pconst = pooled_median[tgt]
    berr, perr = np.abs(actual - bconst), np.abs(actual - pconst)

    pd.DataFrame(dict(
        source_project=src, target_project=tgt,
        pdf_source="", pdf_target="",
        issue_id=mt.issue_id.values[te], issue_key=mt.issue_key.values[te],
        actual_story_point=actual, predicted_story_point=pred, absolute_error=err,
        baseline_prediction=bconst, baseline_absolute_error=berr,
        pooled_median_prediction=pconst, pooled_median_absolute_error=perr,
    )).to_csv(f"{PRED_DIR}/{safe(src)}__{safe(tgt)}.csv", index=False)
    del mdl
    if DEV == "cuda":
        torch.cuda.empty_cache()

    return dict(val_mae_mean=float(np.mean(fold_vals)), val_mae_std=float(np.std(fold_vals)),
                val_best_epoch_mean=float(np.mean(best_epochs)),
                val_best_epoch_min=int(np.min(best_epochs)), val_best_epoch_max=int(np.max(best_epochs)),
                epochs=fix_epochs, test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                baseline_mae=float(berr.mean()), baseline_mdae=float(np.median(berr)),
                pooled_median_mae=float(perr.mean()), pooled_median_mdae=float(np.median(perr)),
                n_train=int(len(tr)), n_test=int(len(te)))


rows_out = []
for i, (s, t) in enumerate(PAIRS):
    src, tgt = dsn(s), dsn(t)
    t_pair = time.time()
    print(f"[{i+1}/39] {s} -> {t}", flush=True)
    try:
        r = run_pair(src, tgt)
        status = "ok"
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"[{i+1}/39] FAILED: {type(ex).__name__}: {ex}", flush=True)
        r = {k: float("nan") for k in ("val_mae_mean", "val_mae_std", "val_best_epoch_mean",
                                       "test_mae", "test_mdae", "baseline_mae", "baseline_mdae",
                                       "pooled_median_mae", "pooled_median_mdae")}
        r.update(epochs=0, n_train=0, n_test=0)
        status = f"failed:{type(ex).__name__}"
    r.update(pdf_source=s, pdf_target=t, source=src, target=tgt, status=status,
             seconds=round(time.time() - t_pair, 1))
    rows_out.append(r)
    pd.DataFrame(rows_out).to_csv(f"{OUT}/results_pairs{SUF}.csv", index=False)
    if status == "ok":
        print(f"    test MAE {r['test_mae']:.3f} (baseline {r['baseline_mae']:.3f}, "
              f"pooled median {r['pooled_median_mae']:.3f}) | MdAE {r['test_mdae']:.3f} | "
              f"val {r['val_mae_mean']:.3f}+/-{r['val_mae_std']:.3f} -> {r['epochs']} ep | "
              f"{time.time() - t_start:.0f}s total", flush=True)
print("ALL PAIRS DONE in", round((time.time() - t_start) / 60, 1), "min")

# %% [markdown]
# ## 8. Summary and paired significance tests
#
# Macro (pair-average) and issue-weighted MAE answer different questions; both are reported.
# The pooled MdAE is the median over all held-out issues, not the mean of 39 per-pair medians.
# Wilcoxon tests are paired at issue level over the very same held-out rows.

# %%
from scipy.stats import wilcoxon

res = pd.DataFrame(rows_out)
ok = res[res.status == "ok"].copy()
pred_files = sorted(glob.glob(f"{PRED_DIR}/*.csv"))
preds = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
preds.to_csv(f"{OUT}/predictions_all{SUF}.csv", index=False)

w_m = wilcoxon(preds.absolute_error, preds.baseline_absolute_error)
w_p = wilcoxon(preds.absolute_error, preds.pooled_median_absolute_error)
w_pair = wilcoxon(ok.test_mae, ok.baseline_mae)

summary = [
    f"PS-3 ASSIGNED EXPERIMENT — BERT word embeddings + 3-layer CNN | tag {TAG}",
    f"seed {SEED} | story points ({SP_MIN},{SP_MAX}] | BERT window {MAX_LEN} tokens | "
    f"{N_FOLDS}-fold CV on source development pool | single held-out target test",
    f"corpus: {len(df)} issues / {df.project.nunique()} projects | frozen bert-base-uncased, fp16 arrays",
    "",
    f"pairs evaluated            : {len(ok)} / {len(res)}",
    f"macro test MAE             : {ok.test_mae.mean():.3f} +/- {ok.test_mae.std():.3f}",
    f"macro baseline MAE         : {ok.baseline_mae.mean():.3f} +/- {ok.baseline_mae.std():.3f}",
    f"macro pooled-median MAE    : {ok.pooled_median_mae.mean():.3f}",
    f"macro MAE improvement      : {100 * (1 - ok.test_mae.mean() / ok.baseline_mae.mean()):+.1f}% vs single-source median",
    f"issue-weighted test MAE    : {float(np.average(ok.test_mae, weights=ok.n_test)):.3f}",
    f"issue-weighted baseline MAE: {float(np.average(ok.baseline_mae, weights=ok.n_test)):.3f}",
    f"pooled over {len(preds)} issues: MAE {preds.absolute_error.mean():.3f} | MdAE {preds.absolute_error.median():.3f}",
    f"mean per-pair test MdAE    : {ok.test_mdae.mean():.3f} (baseline {ok.baseline_mdae.mean():.3f}, "
    f"pooled median {ok.pooled_median_mdae.mean():.3f})",
    f"5-fold validation MAE      : {ok.val_mae_mean.mean():.3f} +/- {ok.val_mae_std.mean():.3f} (best epoch)",
    f"final epochs               : mean {ok.epochs.mean():.1f} (min {int(ok.epochs.min())}, max {int(ok.epochs.max())})",
    f"beats single-source median : {int((ok.test_mae < ok.baseline_mae).sum())}/{len(ok)} pairs",
    f"beats pooled median        : {int((ok.test_mae < ok.pooled_median_mae).sum())}/{len(ok)} pairs",
    f"unbeatable baseline (MAE=0): {int((ok.baseline_mae == 0).sum())} pairs",
    f"Wilcoxon issue-level vs single-source median : p = {w_m.pvalue:.3e}",
    f"Wilcoxon issue-level vs pooled median        : p = {w_p.pvalue:.3e}",
    f"Wilcoxon pair-level (39 pairs)               : p = {w_pair.pvalue:.4f}",
    "",
    "best 5 pairs:",
    ok.nsmallest(5, "test_mae")[["pdf_source", "pdf_target", "test_mae", "test_mdae",
                                 "baseline_mae", "n_test"]].to_string(index=False),
    "",
    "worst 5 pairs:",
    ok.nlargest(5, "test_mae")[["pdf_source", "pdf_target", "test_mae", "test_mdae",
                                "baseline_mae", "n_test"]].to_string(index=False),
]
txt = "\n".join(summary)
open(f"{OUT}/evaluation_summary{SUF}.txt", "w").write(txt)
print(txt)

master = dict(
    tag=TAG,
    configuration=f"{MAX_LEN} tokens / story points ({SP_MIN},{SP_MAX}]",
    window=("full stored 256-token arrays" if MAX_LEN == STORED_LEN
            else f"first {MAX_LEN} positions of the stored 256-token arrays (controlled ablation)"),
    model="per-source BERT + 3-layer CNN", pairs=int(len(ok)),
    macro_test_mae=float(ok.test_mae.mean()), macro_test_mae_std=float(ok.test_mae.std()),
    macro_baseline_mae=float(ok.baseline_mae.mean()),
    macro_pooled_median_mae=float(ok.pooled_median_mae.mean()),
    mean_test_mdae=float(ok.test_mdae.mean()), mean_baseline_mdae=float(ok.baseline_mdae.mean()),
    issue_weighted_mae=float(np.average(ok.test_mae, weights=ok.n_test)),
    issue_weighted_baseline_mae=float(np.average(ok.baseline_mae, weights=ok.n_test)),
    pooled_issue_mae=float(preds.absolute_error.mean()),
    pooled_issue_mdae=float(preds.absolute_error.median()),
    val_mae_mean=float(ok.val_mae_mean.mean()), val_mae_std=float(ok.val_mae_std.mean()),
    beats_baseline=int((ok.test_mae < ok.baseline_mae).sum()),
    beats_pooled_median=int((ok.test_mae < ok.pooled_median_mae).sum()),
    wilcoxon_issue_vs_baseline=float(w_m.pvalue),
    wilcoxon_issue_vs_pooled_median=float(w_p.pvalue),
    wilcoxon_pair=float(w_pair.pvalue),
    held_out_issues=int(len(preds)),
)
json.dump(master, open(f"{OUT}/master_metrics{SUF}.json", "w"), indent=2)

# %% [markdown]
# ## 9. Plots

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# per-pair comparison
plot = ok.sort_values("test_mae")
labels = [f"{s} -> {t}" for s, t in zip(plot.pdf_source, plot.pdf_target)]
ypos = np.arange(len(plot))
fig, ax = plt.subplots(figsize=(11, 13))
ax.barh(ypos + 0.2, plot.test_mae, 0.4, color="#2b6cb0", label="BERT + CNN (per source)")
ax.barh(ypos - 0.2, plot.baseline_mae, 0.4, color="#cbd5e0", edgecolor="#718096",
        label="single-source median baseline")
ax.set_yticks(ypos, labels, fontsize=7.5)
ax.axvline(ok.test_mae.mean(), color="#2b6cb0", ls=":", lw=1.2, label=f"model macro {ok.test_mae.mean():.2f}")
ax.axvline(ok.baseline_mae.mean(), color="#4a5568", ls="--", lw=1.2,
           label=f"baseline macro {ok.baseline_mae.mean():.2f}")
ax.invert_yaxis()
ax.set_xlabel("test MAE (story points)")
ax.set_title(f"PS-3 assigned pairs — {MAX_LEN} tokens, story points ({SP_MIN},{SP_MAX}] "
             f"(tag {TAG})\n"
             "per-source BERT + CNN vs median baseline, seed 42, 5-fold protocol")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/results_per_pair{SUF}.png")
plt.close()

# scatter
fig, ax = plt.subplots(figsize=(6, 6))
lim = max(ok.test_mae.max(), ok.baseline_mae.max()) * 1.05
ax.plot([0, lim], [0, lim], color="#a0aec0", ls="--", lw=1)
ax.scatter(ok.baseline_mae, ok.test_mae, s=28, color="#2b6cb0", alpha=0.8)
ax.set_xlabel("baseline test MAE")
ax.set_ylabel("BERT + CNN test MAE")
ax.set_title("Per-pair test MAE: model vs median baseline")
plt.tight_layout()
plt.savefig(f"{OUT}/scatter{SUF}.png")
plt.close()

# error distributions
fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(0, min(20, np.percentile(preds.absolute_error, 99)), 60)
ax.hist(preds.baseline_absolute_error, bins=bins, alpha=0.6, label="median baseline", color="#a0aec0")
ax.hist(preds.absolute_error, bins=bins, alpha=0.6, label="BERT + CNN", color="#2b6cb0")
ax.set_yscale("log")
ax.set_xlabel("absolute error (story points)")
ax.set_ylabel("issues (log scale)")
ax.set_title(f"Held-out error distribution, {len(preds)} issues")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/error_distribution{SUF}.png")
plt.close()

# token length / truncation
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(np.clip(L, 0, 800), bins=100, color="#2b6cb0")
ax.axvline(128, color="#dd6b20", ls="--", label=f"128 → {tstats['trunc_pct_128']:.1f}% truncated")
ax.axvline(256, color="#38a169", ls="--", label=f"256 → {tstats['trunc_pct_256']:.1f}% truncated")
ax.set_xlabel("untruncated BERT token length")
ax.set_ylabel("issues")
ax.set_title(f"Untruncated token lengths — {MAX_LEN}-token window in use (tag {TAG})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/token_lengths{SUF}.png")
plt.close()

# story point distribution, before/after filter
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(np.clip(sp[sp <= 100], 0, 100), bins=50, color="#805ad5", alpha=0.85)
ax.set_xlabel("story point")
ax.set_ylabel("issues")
ax.set_title("Story-point distribution after the (0,100] filter")
plt.tight_layout()
plt.savefig(f"{OUT}/storypoint_distribution{SUF}.png")
plt.close()

# %% [markdown]
# ## 10. Reproducibility record

# %%
json.dump({
    "experiment": f"PS-3 assigned cross-project random-split evaluation (redo, tag {TAG})",
    "tag": TAG,
    "configuration": {"seed": SEED, "split": "80:20 random per project", "folds": N_FOLDS,
                      "story_point_filter": f"({SP_MIN},{SP_MAX}]", "max_length": MAX_LEN,
                      "stored_array_length": STORED_LEN,
                      "window_note": ("full stored arrays" if MAX_LEN == STORED_LEN else
                                      "first MAX_LEN positions of the stored arrays; [SEP] of long "
                                      "issues is not present in this controlled view"),
                      "embeddings": "bert-base-uncased frozen last hidden state, float16",
                      "embedding_sources": ["madmaxop123/tawos-bert-256-a",
                                            "madmaxop123/tawos-bert-256-b"],
                      "model": "3-layer 1D CNN 768-256-128-64, kernel 3, masked max pool, Softplus head",
                      "loss": "L1", "optimizer": "Adam", "lr": LR, "batch_size": BS,
                      "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
                      "epoch_selection": "mean of the 5 best validation epochs"},
    "corpus": audit, "truncation": tstats,
    "device": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
    "versions": {"python": __import__("sys").version.split()[0], "torch": torch.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__,
                 "sklearn": __import__("sklearn").__version__},
    "runtime_minutes": round((time.time() - t_start) / 60, 1),
}, open(f"{OUT}/run_metadata{SUF}.json", "w"), indent=2)
print("DONE. outputs:", sorted(os.listdir(OUT)))
