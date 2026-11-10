# 12_ps3_256_ablation.py
#
# The redo configuration: assigned PS-3 protocol at story points in (0, 100] and a 256-token
# window, with the predictor ablation run under the same protocol instead of pooled training.

# %% [markdown]
# # 12 — PS-3 Redo: (0, 100] Labels, 256-Token Window, CNN vs Transformer vs MLP
#
# This is the follow-up configuration requested after the main evaluation. **Two things change
# at once relative to 05/07, and this kernel is the only place they are measured together:**
#
# 1. **Label filter (0, 40] → (0, 100]** — the standard Deep-SE cap (Choetkiertikul et al. 2018):
#    drop SP ≤ 0 and SP > 100. 62,870 issues instead of 62,193 (the 141 values above 40 and ≤ 100
#    return; the corpus' largest legitimate label is now 100, not 40). The result is a separate
#    configuration, not a drop-in replacement: its numbers are comparable with 05 only through
#    the in-kernel baseline, which is recomputed on the same (0, 100] partitions.
# 2. **Token window 128 → 256** — the measured truncation fix. On this corpus 29.5% of issues
#    exceed 128 tokens (41.8% of token content lost, 92.9% of Jira Cloud); at 256 the truncated
#    share drops to ~9% (11a/11b print the exact per-half numbers).
#
# **Protocol = the assigned PS-3 protocol, unchanged in every other respect:** per assigned pair,
# 5-fold cross-validation on the source project's development pool with best-epoch selection, a
# final model trained on the full pool for the mean best epoch, and one evaluation on the target's
# held-out 20%. Seed 42, random 80:20 per project. The median baseline is recomputed in-kernel on
# these partitions (both the per-source constant and the within-project reference).
#
# **Predictor ablation under the same protocol** (guideline §3.1 allows a 3-layer CNN; 07 showed
# the choice barely matters pooled — this re-tests that per-pair at 256 tokens):
#
# | Head | Architecture |
# |---|---|
# | `cnn` | 3-layer 1D CNN 768→256→128→64, kernel 3, masked max-pool, 64→32→1 Softplus (the 05 model) |
# | `transformer` | linear proj to d=256, 1 self-attention layer (4 heads, FFN 512), masked mean, 256→32→1 (the 07 model) |
# | `mlp` | masked mean over real tokens, 768→256→32→1 (the 07 control) |
#
# All heads: L1 loss (matches MAE), Adam lr 1e-3, batch 64, patience 4, max 25 epochs, raw
# story-point target. Per-pair issue-level predictions are saved for the extended metrics:
# per pair — MAE, MdAE, **SA** (1000 seeded random guesses from the source train pool),
# **Vargha–Delaney Â12** vs the median baseline's errors, and **scale-normalized MAE**
# (MAE / target train-pool median). Per-head Wilcoxon tests against the baseline and head-vs-head
# close the summary.
#
# **Inputs:** dataset `tawos-extracted`; kernel outputs `tawos-bert-256-a`, `tawos-bert-256-b`
# (`emb256/`, `meta/`). **Needs GPU** (~4–5 h on a T4: 39 pairs × 3 heads × (5 CV folds + 1 final)
# at 256 tokens).
# **Outputs:** `results_baseline_100.csv`, `results_ps3_256.csv`, `predictions_ps3_256/`,
# `ps3_256_summary.txt`, `ps3_256_per_pair.png`.

# %%
import glob
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.model_selection import KFold

SEED = 42
MAX_LEN = 256
EMB_WIDTH = 768
BS = 64
LR = 1e-3
MAX_EPOCHS = 25
PATIENCE = 4
N_FOLDS = 5
SA_DRAWS = 1000
HEADS = ("cnn", "transformer", "mlp")

csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 100)]        # the redo label filter
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()
print(f"{len(df)} issues | {df.project.nunique()} projects | filter (0, 100]")

rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    labels = np.array(g.index)                  # copy: safe to shuffle in place
    rank = {lab: i for i, lab in enumerate(labels)}
    rng.shuffle(labels)
    n_test = max(1, int(round(0.2 * len(labels))))
    splits[proj] = {"test": np.array([rank[v] for v in labels[:n_test]]),
                    "train": np.array([rank[v] for v in labels[n_test:]])}

# One construction point: unlike 02→05 there is no committed (0,100] baseline to verify against,
# so this kernel builds the baseline and the partitions together and everything downstream reads
# the same objects. The baseline CSV is still written out for the repo and the report.
grouped = {p: df[df.project == p].reset_index(drop=True) for p in sorted(df.project.unique())}
within_rows, cross_rows = [], []
for p, g in grouped.items():
    y_te = g.storypoint.values[splits[p]["test"]]
    med = float(np.median(g.storypoint.values[splits[p]["train"]]))
    err = np.abs(y_te - med)
    within_rows.append(dict(source=p, target=f"{p} [WITHIN]", mae=float(err.mean()),
                            mdae=float(np.median(err)), pred=med,
                            n_train=len(splits[p]["train"]), n_test=len(y_te)))
    for q, h in grouped.items():
        if q == p:
            continue
        yq = h.storypoint.values[splits[q]["test"]]
        eq = np.abs(yq - med)
        cross_rows.append(dict(source=p, target=q, mae=float(eq.mean()), mdae=float(np.median(eq)),
                               pred=med, n_train=len(splits[p]["train"]), n_test=len(yq)))
base = pd.DataFrame(cross_rows + within_rows)
base.to_csv("/kaggle/working/results_baseline_100.csv", index=False)
within = pd.DataFrame(within_rows).set_index("source")
cross = pd.DataFrame(cross_rows)
print(f"baseline on (0,100]: cross-project macro MAE {cross.mae.mean():.3f} +/- {cross.mae.std():.3f} "
      f"| within-project {within.mae.mean():.3f} | {len(cross)} cross pairs")

# %% [markdown]
# ## Loading the 256-token embeddings (two kernel halves)

# %%


def find_dir(*patterns):
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    raise AssertionError(f"no input directory matching {patterns}")


EMB_A = find_dir("/kaggle/input/tawos-bert-256-a/emb256", "/kaggle/input/**/emb256")
EMB_B = find_dir("/kaggle/input/tawos-bert-256-b/emb256")
print("half a:", EMB_A, "\nhalf b:", EMB_B)


def safe(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


_cache = {}


def load_project(name):
    """Merged 256-token arrays (mmap) + metadata for one project, from whichever half holds it."""
    if name not in _cache:
        hits = [d for d in (EMB_A, EMB_B)
                if os.path.exists(os.path.join(d, safe(name) + ".npy"))
                or os.path.exists(os.path.join(d, safe(name).rstrip("_") + ".npy"))]
        assert len(hits) == 1, f"{name}: found in {len(hits)} halves"
        d = hits[0]
        key = safe(name) if os.path.exists(os.path.join(d, safe(name) + ".npy")) \
            else safe(name).rstrip("_")
        meta = pd.read_csv(os.path.join(os.path.dirname(d), "meta", key + ".csv"))
        g = grouped[name]
        assert len(meta) == len(g), f"{name}: {len(meta)} meta rows vs {len(g)} filtered rows"
        assert (meta.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), \
            f"{name}: rows not aligned with the filtered frame"
        emb = np.load(os.path.join(d, key + ".npy"), mmap_mode="r")
        assert emb.shape == (len(meta), MAX_LEN, EMB_WIDTH), (name, emb.shape)
        _cache[name] = (emb, meta, np.clip(meta.n_tokens.values, 1, MAX_LEN).astype(np.int64))
    return _cache[name]

# %% [markdown]
# ## The 39 assigned pairs (same list and mappings as 05/06/08)

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
# ## Models
#
# The three predictors consume the same `[batch, 256, 768]` token blocks with the same token mask
# and end in the same Softplus head. The CNN keeps 05's masking discipline (pads zeroed before the
# first convolution and after every layer, excluded from the max pool); the Transformer uses a
# `src_key_padding_mask`; the MLP masks the mean explicitly.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class CNNRegressor(nn.Module):
    """The 05 predictor: three masked convolutions, masked global max pool, small Softplus head."""

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


class TransformerRegressor(nn.Module):
    """The 07 transformer: project to d=256, one self-attention layer, masked mean, Softplus head."""

    def __init__(self, d_in=EMB_WIDTH, d_model=256, nhead=4, ff=512, p=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=p,
                                           activation="relu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x, ntok):
        keep = torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]
        h = self.proj(self.drop(x))
        h = self.encoder(h, src_key_padding_mask=~keep)
        m = keep.unsqueeze(2).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.softplus(self.fc2(F.relu(self.fc1(pooled))).squeeze(-1))


class MLPRegressor(nn.Module):
    """The 07 control: masked mean over the real tokens, no convolution, no attention."""

    def __init__(self, d_in=EMB_WIDTH, hidden=256, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, ntok):
        keep = (torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]).unsqueeze(2).float()
        pooled = (self.drop(x) * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return F.softplus(self.fc3(F.relu(self.fc2(F.relu(self.fc1(pooled)))))).squeeze(-1)


MODELS = {"cnn": CNNRegressor, "transformer": TransformerRegressor, "mlp": MLPRegressor}

print("parameter budget:",
      {h: round(sum(p.numel() for p in MODELS[h]().parameters()) / 1e6, 2) for h in HEADS}, "M")


def make_batch(head, prj, idx, y):
    emb, meta, ntok = load_project(prj)
    x = torch.tensor(np.asarray(emb[idx]), dtype=torch.float32, device=DEV)
    nt = torch.tensor(ntok[idx], dtype=torch.long, device=DEV)
    yv = torch.tensor(y, dtype=torch.float32, device=DEV)
    return (x, nt), yv


@torch.no_grad()
def predict(model, head, prj, idx, bs=256):
    model.eval()
    out = []
    for b0 in range(0, len(idx), bs):
        args, _ = make_batch(head, prj, idx[b0:b0 + bs], np.zeros(min(bs, len(idx) - b0)))
        out.append(model(*args).float().cpu().numpy())
    return np.concatenate(out)


def train_model(head, prj, pool, ypool, epochs=MAX_EPOCHS, bs=BS, lr=LR, patience=PATIENCE,
                seed=0, va=None):
    """Train one head on `pool` (per-project row positions); with `va`, best-epoch weights are
    restored and (model, hist, best_val_mae, best_epoch) returned."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[head]().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.L1Loss()
    best_val, best_epoch, best_state = None, None, None
    bad_epochs, hist = 0, []
    va_idx, va_y = va if va is not None else (None, None)
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(pool))
        for b0 in range(0, len(pool), bs):
            sel = pool[perm[b0:b0 + bs]]
            args, yv = make_batch(head, prj, sel, ypool[perm[b0:b0 + bs]])
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(*args), yv)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if va_idx is None:
            continue
        vmae = float(np.abs(predict(model, head, prj, va_idx) - va_y).mean())
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
# ## Run the 39 pairs × 3 heads
#
# Per pair and head: 5-fold CV on the source pool (best-epoch selection), a final model trained on
# the full pool for the mean best epoch, one evaluation on the target's held-out 20%. The results
# CSV is rewritten after every pair, so an interrupted session still leaves everything finished.

# %%
os.makedirs("/kaggle/working/predictions_ps3_256", exist_ok=True)
base_idx = base.set_index(["source", "target"])
rows, t0 = [], time.time()
for head in HEADS:
    for i, (s, t) in enumerate(PAIRS):
        src, tgt = dsn(s), dsn(t)
        gs, gt = grouped[src], grouped[tgt]
        tr_pos, te_pos = splits[src]["train"], splits[tgt]["test"]
        y_src = gs.storypoint.values.astype(np.float32)
        y_te = gt.storypoint.values[te_pos].astype(np.float32)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_vals, best_epochs = [], []
        for f, (a, b) in enumerate(kf.split(np.arange(len(tr_pos)))):
            mdl, hist, vmae, vep = train_model(head, src, tr_pos[a], y_src[tr_pos[a]],
                                               seed=SEED + f, va=(tr_pos[b], y_src[tr_pos[b]]))
            fold_vals.append(vmae)
            best_epochs.append(vep)
            del mdl
            if DEV == "cuda":
                torch.cuda.empty_cache()
        fix_epochs = int(np.clip(round(float(np.mean(best_epochs))), 1, MAX_EPOCHS))
        mdl, _, _, _ = train_model(head, src, tr_pos, y_src, epochs=fix_epochs, seed=SEED)
        pred = predict(mdl, head, tgt, te_pos)
        del mdl
        if DEV == "cuda":
            torch.cuda.empty_cache()

        err = np.abs(pred - y_te)
        pred_const = float(base_idx.loc[(src, tgt)].pred)
        pd.DataFrame({
            "head": head, "source_project": src, "target_project": tgt,
            "issue_id": gt.issue_id.values[te_pos], "issue_key": gt.issue_key.values[te_pos],
            "actual_story_point": y_te, "predicted_story_point": pred, "absolute_error": err,
            "source_median_prediction": pred_const,
        }).to_csv(f"/kaggle/working/predictions_ps3_256/{head}__{safe(src)}__{safe(tgt)}.csv",
                  index=False)

        brow = base_idx.loc[(src, tgt)]
        rows.append(dict(head=head, pdf_source=s, pdf_target=t, source=src, target=tgt,
                         val_mae_mean=float(np.mean(fold_vals)),
                         val_mae_std=float(np.std(fold_vals)),
                         epochs=fix_epochs, n_test=int(len(y_te)),
                         test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                         baseline_mae=float(brow.mae), baseline_mdae=float(brow.mdae),
                         beats_baseline=bool(err.mean() < float(brow.mae))))
        pd.DataFrame(rows).to_csv("/kaggle/working/results_ps3_256.csv", index=False)
        print(f"[{head} {i + 1}/39] {s} -> {t}: test MAE {rows[-1]['test_mae']:.3f} "
              f"(baseline {rows[-1]['baseline_mae']:.3f}), MdAE {rows[-1]['test_mdae']:.3f} "
              f"| val {rows[-1]['val_mae_mean']:.3f}+/-{rows[-1]['val_mae_std']:.3f} "
              f"@ep{rows[-1]['epochs']} | {time.time() - t0:.0f}s", flush=True)

# %% [markdown]
# ## Extended metrics and summary
#
# Per head: macro and issue-weighted test MAE, mean per-pair MdAE, pooled MdAE, mean 5-fold
# validation MAE, SA, Â12 vs the baseline's errors, scale-normalized MAE, and per-pair Wilcoxon
# tests (39 pairs are the independent units — issues within one project are not).

# %%
def sa(y_te, pool, mae_model, seed):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pool), size=(SA_DRAWS, len(y_te)))
    return 1.0 - mae_model / float(np.abs(y_te[None, :] - pool[draws]).mean())


def vda(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X == Y); X = baseline errors, Y = head errors."""
    n1, n2 = len(x), len(y)
    r = rankdata(np.concatenate([x, y]))
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


pool_sp = {p: grouped[p].storypoint.values[splits[p]["train"]] for p in grouped}
tgt_median = {p: float(within.loc[p, "pred"]) for p in grouped}
preds = {}
for head in HEADS:
    files = sorted(glob.glob(f"/kaggle/working/predictions_ps3_256/{head}__*.csv"))
    assert len(files) == 39, (head, len(files))
    preds[head] = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

ext_rows = []
for head in HEADS:
    res_h = [r for r in rows if r["head"] == head]
    for i, r in enumerate(res_h):
        g = preds[head][(preds[head].source_project == r["source"]) &
                        (preds[head].target_project == r["target"])]
        assert len(g) == r["n_test"], (head, r["pdf_source"])
        errs = g.absolute_error.values
        base_errs = np.abs(g.actual_story_point.values - g.source_median_prediction.values)
        ext_rows.append(dict(head=head, pdf_source=r["pdf_source"], pdf_target=r["pdf_target"],
                             mae=r["test_mae"], mdae=r["test_mdae"], n_test=r["n_test"],
                             baseline_mae=r["baseline_mae"],
                             sa=sa(g.actual_story_point.values.astype(np.float64),
                                   pool_sp[r["source"]], r["test_mae"], SEED * 1000 + i),
                             a12_vs_baseline=vda(base_errs, errs),
                             norm_mae=r["test_mae"] / tgt_median[r["target"]]))
em = pd.DataFrame(ext_rows)
em.to_csv("/kaggle/working/extended_metrics_ps3_256.csv", index=False)

b_idx = em.set_index(["pdf_source", "pdf_target"])
summary_bits = []
for head in HEADS:
    r = em[em["head"] == head]
    res_h = pd.DataFrame([x for x in rows if x["head"] == head])
    summary_bits.append(dict(
        head=head, macro=r.mae.mean(), macro_std=r.mae.std(), mean_mdae=r.mdae.mean(),
        pooled_mdae=float(np.median(np.concatenate([preds[head].absolute_error.values]))),
        val=res_h.val_mae_mean.mean(), mean_sa=r.sa.mean(), mean_a12=r.a12_vs_baseline.mean(),
        a12_win=int((r.a12_vs_baseline > 0.5).sum()),
        beats_baseline=int((res_h.test_mae < res_h.baseline_mae).sum()),
        mean_norm=r.norm_mae.mean(), mean_epochs=res_h.epochs.mean()))

cross_macro = float(cross.mae.mean())
within_macro = float(within.mae.mean())
lines = [
    "PS-3 redo - (0, 100] labels, 256-token window, assigned protocol, 3 predictors",
    f"seed {SEED} | {len(df)} issues | same 80:20 partitions for every head and the baseline "
    f"(built here, no committed (0,100] baseline exists)",
    f"baseline (source median constant): cross-project macro MAE {cross_macro:.3f} "
    f"+/- {cross.mae.std():.3f} | within-project {within_macro:.3f}",
    "",
    f"{'head':12s} {'valMAE':>7s} {'MAE':>7s} {'MdAE':>7s} {'pMdAE':>7s} {'SA':>7s} {'A12':>7s} "
    f"{'nrmMAE':>7s} {'beats':>7s} {'epochs':>7s}",
    "-" * 90,
]
for sb_ in summary_bits:
    lines.append(f"{sb_['head']:12s} {sb_['val']:>7.3f} {sb_['macro']:>7.3f} {sb_['mean_mdae']:>7.3f} "
                 f"{sb_['pooled_mdae']:>7.3f} {sb_['mean_sa']:>7.3f} {sb_['mean_a12']:>7.3f} "
                 f"{sb_['mean_norm']:>7.3f} {sb_['beats_baseline']:>6d}/39 {sb_['mean_epochs']:>7.1f}")
lines.append("")
b_head = em[em["head"] == "cnn"].set_index(["pdf_source", "pdf_target"])
for head in HEADS:
    h_idx = em[em["head"] == head].set_index(["pdf_source", "pdf_target"])
    p = float(wilcoxon(h_idx.mae, b_idx.loc[h_idx.index, "mae"]).pvalue)
    lines.append(f"per-pair Wilcoxon {head} vs median_baseline: p = {p:.3f}")
for a_head in HEADS[1:]:
    x = em[em["head"] == a_head].set_index(["pdf_source", "pdf_target"]).mae
    y = b_idx.loc[x.index, "mae"]
    p = float(wilcoxon(x, y).pvalue)
    lines.append(f"per-pair Wilcoxon {a_head} vs cnn: p = {p:.3f}")
txt = "\n".join(lines)
open("/kaggle/working/ps3_256_summary.txt", "w").write(txt)
print(txt)

# %% [markdown]
# ## Plot
#
# Per-pair test MAE, one panel row per assigned pair, all three heads against the (0,100] median
# baseline.

# %%
import matplotlib.pyplot as plt

plot = em.copy()
fig, ax = plt.subplots(figsize=(12, 14))
pair_order = [(s, t) for s, t in PAIRS]
ypos = np.arange(len(pair_order))
colors = {"cnn": "tab:blue", "transformer": "tab:green", "mlp": "tab:red", "baseline": "lightgray"}
for k, meth in enumerate(["baseline", "cnn", "transformer", "mlp"]):
    if meth == "baseline":
        vals = [float(base_idx.loc[(dsn(s), dsn(t)), "mae"]) for s, t in pair_order]
    else:
        g = plot[plot["head"] == meth].set_index(["pdf_source", "pdf_target"])
        vals = [g.loc[(s, t), "mae"] for s, t in pair_order]
    ax.barh(ypos + (k - 1.5) * 0.2, vals, 0.2, color=colors[meth], label=meth)
ax.set_yticks(ypos, [f"{s} -> {t}" for s, t in pair_order], fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("test MAE (story points)")
ax.set_title("PS-3 redo on (0, 100] labels with 256-token embeddings\n"
             "assigned protocol, 5-fold epoch selection, same partitions everywhere")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/ps3_256_per_pair.png", dpi=110)
plt.show()
