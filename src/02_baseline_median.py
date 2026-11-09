# 02_baseline_median.py
#
# The floor every model has to beat. CPU only, runs in under a minute.

# %% [markdown]
# # 02 — Median Baseline (cross-project)
#
# The simplest defensible predictor: for a target issue, output the **source project's median story
# point** from its own development pool. No text is read. Every BERT result in `04`–`06` is judged
# against this, per pair.
#
# Two settings are computed:
#
# - **cross-project** — all 39 × 38 = 1,482 ordered source→target pairs (the 39 assigned in the PS-3
#   table are a subset, filtered in `05`);
# - **within-project** — source = target, as the difficulty reference for the same median predictor.
#
# **Input:** dataset `tawos-extracted` (`all_issues_sp.csv` from `01`). **Output:**
# `baseline_results.csv` (per pair MAE, MdAE, the constant prediction, train/test sizes),
# `baseline_summary.txt`, `baseline_by_source.png`. CPU only.
#
# The seed-42 partitions built here are reused by `04`, `05` and `06`, so the model results are
# comparable with these numbers pair by pair. That reuse depends on this loop iterating the projects
# in the same order with the same names, which `05` now verifies instead of assuming.

# %%
import glob

import numpy as np
import pandas as pd

SEED = 42
SP_MIN, SP_MAX = 0, 40

hits = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)
assert hits, "attach the tawos-extracted dataset as Input"
df = pd.read_csv(hits[0])
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
n_raw = len(df)

# Documented outlier policy: story points of 0 (no-effort tickets) and above 40 (corrupt values, the
# largest in the dump is 26,710,114) are excluded from every experiment. Per-project counts are in
# 01's filter_impact.csv.
df = df[(df.storypoint > SP_MIN) & (df.storypoint <= SP_MAX)]
df["project"] = df["project"].astype(str).str.strip()
print(f"{n_raw} -> {len(df)} issues after SP in ({SP_MIN}, {SP_MAX}]")

rng = np.random.default_rng(SEED)
splits = {}
for proj, g in df.groupby("project"):
    # np.array(g.index) copies: below pandas 3, to_numpy() shares the Index buffer with the
    # shuffle below. The label sets are the same either way, but the copy keeps this notebook's
    # draws identical to 04/05/06, which rebuild the very same partitions from the same seed.
    idx = np.array(g.index)
    rng.shuffle(idx)
    n_test = max(1, int(round(0.2 * len(idx))))
    splits[proj] = {"test": idx[:n_test], "train": idx[n_test:]}

projects = sorted(splits)
medians = {p: float(df.loc[splits[p]["train"], "storypoint"].median()) for p in projects}
print(len(projects), "projects | example medians:", {p: medians[p] for p in projects[:5]})

# %% [markdown]
# ## Every pair
#
# `pred` records the constant the baseline emitted, which is the quickest way to see why a pair fails:
# when the source median and the target's own median are far apart, no text model can recover the
# target's scale from a source it has never calibrated against.

# %%
rows = []
for src in projects:
    for tgt in projects:
        if tgt == src:
            continue
        y = df.loc[splits[tgt]["test"], "storypoint"].to_numpy()
        err = np.abs(y - medians[src])
        rows.append(dict(source=src, target=tgt, mae=float(err.mean()), mdae=float(np.median(err)),
                         pred=medians[src], n_train=len(splits[src]["train"]), n_test=len(y)))

for p in projects:
    y = df.loc[splits[p]["test"], "storypoint"].to_numpy()
    err = np.abs(y - medians[p])
    rows.append(dict(source=p, target=f"{p} [WITHIN]", mae=float(err.mean()),
                     mdae=float(np.median(err)), pred=medians[p],
                     n_train=len(splits[p]["train"]), n_test=len(y)))

res = pd.DataFrame(rows).sort_values(["source", "target"])
res.to_csv("/kaggle/working/baseline_results.csv", index=False)

cross = res[~res.target.str.contains("WITHIN")]
within = res[res.target.str.contains("WITHIN")]
summary = [
    f"seed {SEED} | filter SP in ({SP_MIN},{SP_MAX}] | {len(df)} issues | {len(projects)} projects",
    f"CROSS-PROJECT  (n={len(cross)} pairs): MAE mean {cross.mae.mean():.3f} +- {cross.mae.std():.3f}"
    f" | MdAE mean {cross.mdae.mean():.3f}",
    f"WITHIN-PROJECT (n={len(within)}):      MAE mean {within.mae.mean():.3f} +- {within.mae.std():.3f}"
    f" | MdAE mean {within.mdae.mean():.3f}",
    f"gap: cross-project is {100 * (cross.mae.mean() / within.mae.mean() - 1):.0f}% worse than "
    f"within-project with the same predictor",
    f"pairs with an unbeatable baseline (MAE == 0): {int((cross.mae == 0).sum())}",
]
txt = "\n".join(summary)
open("/kaggle/working/baseline_summary.txt", "w").write(txt)
print(txt)
print("\nbest 5 cross pairs:")
print(cross.nsmallest(5, "mae")[["source", "target", "mae", "mdae"]].to_string(index=False))
print("\nworst 5 cross pairs:")
print(cross.nlargest(5, "mae")[["source", "target", "mae", "mdae"]].to_string(index=False))

# %% [markdown]
# ## Which source projects are easy?
#
# A median predictor's error is set entirely by how far the source's scale sits from the target's, so
# this plot is really a picture of scale mismatch across the dataset.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(18, 5))
cross.groupby("source").mae.mean().sort_values().plot.bar(ax=ax, color="steelblue")
ax.axhline(cross.mae.mean(), color="red", ls="--", label=f"overall {cross.mae.mean():.2f}")
ax.set_title("Median-baseline MAE by SOURCE project (lower = easier source)")
ax.set_ylabel("MAE")
ax.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/baseline_by_source.png", dpi=120)
plt.show()

# %% [markdown]
# ## Reading the numbers
#
# - These MAEs are the **floor** every BERT + CNN result must beat, pair by pair.
# - The within-project vs cross-project gap is the cost of transferring at all — quote both.
# - A baseline MAE of exactly 0 (a project whose test issues are all one story point) is a pair no
#   model can win, which matters when counting "pairs where the model beats the baseline".
