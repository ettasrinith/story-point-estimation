# 07_architecture_ablation.py
#
# Architecture ablation over the same frozen BERT token embeddings: is the three-layer CNN the right
# predictor, or would a small Transformer encoder / a plain masked-mean MLP do better on the pooled
# multi-source task? Same protocol as 06 — pooled sources, log1p target, 3-seed ensemble, grouped CV
# for target-blind epoch selection, target-blind out-of-fold calibration, both constant baselines.

# %% [markdown]
# # 07 — Predictor Architecture Ablation (Transformer / MLP vs the CNN of 05-06)
#
# The guidelines allow "a three-layer Convolutional Neural Network" as the prediction component —
# *may*, not *must* — so this notebook holds the representation fixed (frozen `bert-base-uncased`
# token embeddings, exactly the ones `05`/`06` consume) and varies only the predictor:
#
# | Predictor | Pooling | Extra capacity |
# |---|---|---|
# | 3-layer CNN (05/06) | masked global max | convolutions see local token n-grams |
# | **Transformer encoder (this notebook)** | masked mean over real tokens | 1 self-attention layer, d=256, 4 heads, FFN 512 — every token can attend to every other |
# | **MLP (this notebook)** | masked mean over real tokens | none — mean-pooled BERT vectors straight into an MLP |
#
# All three have a comparable parameter budget (~0.2-0.9 M) and end in the same Softplus head. The
# MLP row answers the reviewer question "are the convolutions learning anything, or would pooled
# BERT vectors do?"; the Transformer row answers "is local-token convolution leaving signal on the
# table?".
#
# **Protocol (identical to 06, PS-3-compliant):** for each target, training uses the development
# pools of the **other 38 source projects only**; epoch counts come from project-grouped
# cross-validation with the target excluded from its own average; the affine calibration `a·p + b`
# is fitted on that architecture's out-of-fold predictions of the other 38 projects only; the
# target's held-out 20% test rows are only ever read to score. 3-seed ensemble per architecture.
#
# **Inputs:** dataset `tawos-extracted`, kernel output `tawos-bert-embeddings`. **Needs GPU**
# (~3-4 h on a T4: two architectures × {grouped CV + 39 targets × 3 seeds × ≤8 epochs}) and
# ~13 GB RAM for the pooled fp16 array. Results are rewritten after every pair.

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
CV_MAX_EPOCHS = 10
FINAL_EPOCH_CAP = 8
ENSEMBLE_SEEDS = (42, 43, 44)
N_FOLDS = 5
CV_TRAIN_CAP = 30000
ARCHS = ("transformer", "mlp")          # the CNN row comes from 06's committed run

csv = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)[0]
df = pd.read_csv(csv)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df["project"] = df["project"].astype(str).str.strip()
df["pos"] = np.arange(len(df))

rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    labels = np.array(g.index)                  # copy: safe to shuffle in place
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
    assert (meta.issue_id.astype(str).values == g.issue_id.astype(str).values).all(), \
        f"{proj}: embedding rows not aligned with all_issues_sp.csv"
    assert np.allclose(meta.storypoint.values, g.storypoint.values), f"{proj}: storypoint mismatch"
    gp = g.pos.values
    BIG[gp] = np.asarray(emb)
    NTK[gp] = np.clip(meta.n_tokens.values, 1, MAX_LEN)
    POS[proj] = gp
    del emb
print(f"aligned {len(names)} projects | pooled array {BIG.shape} ({BIG.nbytes / 1e9:.1f} GB)")

pool_by_target = {t: np.concatenate([POS[p][splits[p]["train"]] for p in names if p != t])
                  for t in names}
pool_median_by_target = {t: float(np.median(Y[pool_by_target[t]])) for t in names}
train_all = np.concatenate([POS[p][splits[p]["train"]] for p in names])
GROUP = df.project.values
print(f"pooled train rows: {len(train_all)} | mean pool per target: "
      f"{int(np.mean([len(v) for v in pool_by_target.values()]))}")

base = pd.read_csv(glob.glob("/kaggle/input/**/baseline_results.csv", recursive=True)[0])
base["source"] = base["source"].astype(str).str.strip()
base["target"] = base["target"].astype(str).str.strip()
within = base[base.target.str.contains("WITHIN")].set_index("source")
for proj in splits:
    pool = df[df.project == proj].storypoint.values[splits[proj]["train"]]
    med = float(np.median(pool))
    assert abs(med - float(within.loc[proj, "pred"])) < 1e-6, (proj, med, within.loc[proj, "pred"])
print("splits reproduce the baseline partitions for all 39 projects")

# %% [markdown]
# ## Predictor architectures
#
# All three consume the same `[batch, 128, 768]` frozen BERT tokens with the same token mask and end
# in the same 64→32→1-style Softplus head. The Transformer gets a `src_key_padding_mask` so padded
# positions are invisible to attention; the MLP and both pooling operations mask explicitly.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerRegressor(nn.Module):
    """Project frozen tokens to d=256, one self-attention encoder layer, masked mean pool, MLP head."""

    def __init__(self, d_in=EMB_WIDTH, d_model=256, nhead=4, n_layers=1, ff=512, p=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, dropout=p,
                                           activation="relu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
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
    """Masked mean pool over the frozen token vectors, then a plain MLP — the no-convolution control."""

    def __init__(self, d_in=EMB_WIDTH, hidden=256, p=0.2):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x, ntok):
        keep = (torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]).unsqueeze(2).float()
        pooled = (self.drop(x) * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return F.softplus(self.fc3(F.relu(self.fc2(F.relu(self.fc1(pooled))))).squeeze(-1))


MODELS = {"transformer": TransformerRegressor, "mlp": MLPRegressor}

print("parameter budget:")
for arch, cls in MODELS.items():
    print(f"  {arch:12s} {sum(p.numel() for p in cls().parameters()) / 1e6:.2f} M")


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


def train_one(arch, idx_pool, epochs, bs, seed, va_idx=None):
    """Train one architecture on global row indices `idx_pool`; with `va_idx` keep the
    best-validation-epoch weights and return (model, hist, best_val_mae_in_points, best_epoch)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[arch]().to(DEV)
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
        vmae = float(np.abs(predict_raw(model, va_idx) - Y[va_idx]).mean())
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

# %% [markdown]
# ## Per architecture: grouped CV → target-blind epochs → out-of-fold calibration
#
# Each architecture gets its own project-grouped CV (an attention model may want a different number
# of epochs than a convolution), and its own calibration fitted on its own out-of-fold predictions of
# the other 38 projects. The target's rows never enter its own CV average or its calibration.

# %%
from sklearn.model_selection import GroupKFold

holdout_rows = np.random.RandomState(SEED).choice(train_all, CV_TRAIN_CAP, replace=False)
gkf = GroupKFold(n_splits=N_FOLDS)

ARCH_STATE = {}
for arch in ARCHS:
    fold_records = []
    oof_rows, oof_pred = [], []
    tc0 = time.time()
    for f, (a, b) in enumerate(gkf.split(holdout_rows, groups=df.project.values[holdout_rows])):
        tr, va = holdout_rows[a], holdout_rows[b]
        held_out_projects = set(df.project.values[va].tolist())
        mdl, hist, vmae, vep = train_one(arch, tr, epochs=CV_MAX_EPOCHS, bs=BS,
                                         seed=SEED + f, va_idx=va)
        fold_records.append(dict(fold=f, best_val_mae=vmae, best_epoch=vep, ran_epochs=len(hist),
                                 held_out=held_out_projects))
        oof_rows.append(va)
        oof_pred.append(predict_raw(mdl, va))
        print(f"[{arch}] fold {f + 1}: best val MAE {vmae:.3f} at epoch {vep} of {len(hist)} run "
              f"(holds out {len(held_out_projects)} projects)", flush=True)
        del mdl
        if DEV == "cuda":
            torch.cuda.empty_cache()

    oof_rows = np.concatenate(oof_rows)
    oof_pred = np.concatenate(oof_pred)
    oof_y = Y[oof_rows]
    oof_proj = df.project.values[oof_rows]
    cv_mae = np.array([r["best_val_mae"] for r in fold_records])
    print(f"[{arch}] grouped CV val MAE {cv_mae.mean():.3f} +/- {cv_mae.std():.3f} "
          f"| best epochs {[r['best_epoch'] for r in fold_records]} | {time.time() - tc0:.0f}s",
          flush=True)

    ARCH_STATE[arch] = dict(fold_records=fold_records, oof_rows=oof_rows, oof_pred=oof_pred,
                            oof_y=oof_y, oof_proj=oof_proj, cv_mae=cv_mae)


def epochs_for_target(arch, target):
    """Mean best epoch over this architecture's folds that did NOT hold the target out."""
    recs = ARCH_STATE[arch]["fold_records"]
    use = [r["best_epoch"] for r in recs if target not in r["held_out"]]
    if not use:
        use = [r["best_epoch"] for r in recs]
    return int(np.clip(round(float(np.mean(use))), 1, FINAL_EPOCH_CAP))


def calibration_for_target(arch, target):
    """Affine correction from this architecture's out-of-fold predictions on the OTHER projects.
    The target's own rows (development and test) never enter the fit."""
    st = ARCH_STATE[arch]
    m = st["oof_proj"] != target
    if m.sum() < 100:
        return 1.0, 0.0, float("nan")
    return fit_calibration(st["oof_pred"][m], st["oof_y"][m])

# %% [markdown]
# ## Evaluate the 39 assigned pairs for both architectures

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

os.makedirs("/kaggle/working/predictions_arch", exist_ok=True)
base_idx = base.set_index(["source", "target"])
all_rows = []
t0 = time.time()
for arch in ARCHS:
    rows = []
    out_csv = f"/kaggle/working/results_pooled_{arch}.csv"
    for i, (s, t) in enumerate(PAIRS):
        src, tgt = dsn(s), dsn(t)
        te_pos = POS[tgt][splits[tgt]["test"]]
        ep = epochs_for_target(arch, tgt)
        a, b, oof_before = calibration_for_target(arch, tgt)
        preds = []
        for sd in ENSEMBLE_SEEDS:
            mdl, _, _, _ = train_one(arch, pool_by_target[tgt], epochs=ep, bs=BS, seed=sd)
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
        rows.append(dict(pdf_source=s, pdf_target=t, source=src, target=tgt, architecture=arch,
                         epochs=ep, n_test=int(len(yte)),
                         test_mae=float(err.mean()), test_mdae=float(np.median(err)),
                         cal_mae=float(err_cal.mean()), cal_mdae=float(np.median(err_cal)),
                         cal_a=a, cal_b=b, oof_mae_before_cal=oof_before,
                         baseline_mae=float(brow.mae), baseline_mdae=float(brow.mdae),
                         pooled_median_mae=float(err_pool.mean()),
                         cal_beats_source_baseline=bool(err_cal.mean() < float(brow.mae)),
                         cal_beats_pooled_baseline=bool(err_cal.mean() < float(err_pool.mean()))))
        pd.DataFrame({
            "source_project": src, "target_project": tgt, "architecture": arch,
            "issue_id": df.issue_id.values[te_pos], "actual_story_point": yte,
            "predicted_story_point": pred, "predicted_calibrated": pred_cal,
            "absolute_error": err, "absolute_error_calibrated": err_cal,
        }).to_csv(f"/kaggle/working/predictions_arch/{safe(tgt)}__{arch}.csv", index=False)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{arch} {i + 1}/39] {s} -> {t}: MAE {rows[-1]['test_mae']:.3f} -> cal "
              f"{rows[-1]['cal_mae']:.3f} (a={a:.2f}, b={b:+.2f}; pooled median "
              f"{rows[-1]['pooled_median_mae']:.3f}) @{ep} ep | {time.time() - t0:.0f}s", flush=True)
    all_rows.extend(rows)
    print(f"[{arch}] DONE at {time.time() - t0:.0f}s")
print("ALL DONE in", round((time.time() - t0) / 60, 1), "min")

# %% [markdown]
# ## Summary: does the predictor architecture matter?
#
# The CNN row is quoted from `06`'s run of the identical protocol (same folds, seeds, partitions,
# calibration scheme), so the three predictors are directly comparable. Per-pair Wilcoxon tests say
# whether any architecture difference is real over the 39 pairs.

# %%
res = pd.DataFrame(all_rows)
summary_bits = []
for arch in ARCHS:
    r = res[res.architecture == arch]
    macro = r.cal_mae.mean()
    weighted = float(np.average(r.cal_mae, weights=r.n_test))
    summary_bits.append((arch, macro, weighted,
                         int(r.cal_beats_source_baseline.sum()),
                         int(r.cal_beats_pooled_baseline.sum()),
                         float(r.epochs.mean())))
try:
    from scipy.stats import wilcoxon
    stats_lines = []
    pairs_src = res.pivot(index=["pdf_source", "pdf_target"], columns="architecture",
                          values="cal_mae")
    if "transformer" in pairs_src and "mlp" in pairs_src:
        p_tm = float(wilcoxon(pairs_src["transformer"], pairs_src["mlp"]).pvalue)
        stats_lines.append(f"per-pair Wilcoxon transformer vs mlp: p = {p_tm:.2e}")
    if len(summary_bits) > 1:
        best = min(summary_bits, key=lambda x: x[1])
        for arch, macro, _, _, _, _ in summary_bits:
            if arch == best[0]:
                continue
            r_b = res[res.architecture == best[0]].set_index(["pdf_source", "pdf_target"]).cal_mae
            r_a = res[res.architecture == arch].set_index(["pdf_source", "pdf_target"]).cal_mae
            p = float(wilcoxon(r_b, r_a).pvalue)
            stats_lines.append(f"per-pair Wilcoxon {best[0]} vs {arch}: p = {p:.2e}")
except Exception as ex:
    stats_lines = [f"(significance tests skipped: {type(ex).__name__}: {ex})"]

cnn_line = "(CNN row: see 06's evaluation_summary_pooled.txt — same protocol)"
lines = [
    "Predictor architecture ablation over the same frozen BERT embeddings (pooled protocol, calibrated)",
    f"seed {SEED} | log1p target | ensemble {ENSEMBLE_SEEDS} | grouped CV epochs, target-blind | "
    "calibration on other-38 out-of-fold predictions only",
    "",
] + [
    f"{arch:12s} macro cal MAE {macro:.3f} | issue-weighted {w:.3f} | beats source median "
    f"{bs}/39 | beats pooled median {bp}/39 | mean epochs {ep:.1f}"
    for arch, macro, w, bs, bp, ep in summary_bits
] + [
    "",
    *stats_lines,
    "",
    cnn_line,
]
txt = "\n".join(lines)
open("/kaggle/working/evaluation_summary_arch.txt", "w").write(txt)
print(txt)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 13))
ypos_map = {}
labels = []
for i, (s, t) in enumerate(PAIRS):
    ypos_map[(s, t)] = i
    labels.append(f"{s}  ->  {t}")
styles = {"transformer": ("tab:green", 0.26, 0.0, "Transformer encoder (calibrated)"),
          "mlp": ("tab:red", 0.26, -0.26, "MLP mean-pool control (calibrated)")}
for arch, (color, h, off, lab) in styles.items():
    r = res[res.architecture == arch].set_index(["pdf_source", "pdf_target"]).cal_mae
    ys = [ypos_map[k] + off for k in r.index]
    ax.barh(ys, r.values, h, color=color, label=lab)
pm = {k: v for k, v in zip(res[res.architecture == ARCHS[0]][["pdf_source", "pdf_target"]].itertuples(index=False),
                           res[res.architecture == ARCHS[0]].pooled_median_mae)}
ax.barh([ypos_map[k] - 0.13 for k in pm], list(pm.values()), 0.26, color="tab:orange",
        alpha=0.75, label="pooled median (same 38 projects)")
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("test MAE (story points)")
ax.set_title("Predictor architecture ablation over the same frozen BERT embeddings\n"
             "pooled multi-source protocol, target-blind calibration, 3-seed ensembles")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("/kaggle/working/results_arch_per_pair.png", dpi=110)
plt.show()
