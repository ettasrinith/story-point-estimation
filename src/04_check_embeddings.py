# 04_check_embeddings.py
#
# Standalone integrity check on the 12.2 GB embedding output of 03. CPU only, reads nothing but file
# headers and the first rows of each array.

# %% [markdown]
# # 04 — Verify the Embedding Output
#
# Run this after `03` and before any training. It confirms all 39 arrays exist, each one matches its
# metadata row count, and the stored values are finite — so a failed extraction is caught here rather
# than in the middle of a 39-pair run.
#
# **Input:** kernel output `tawos-bert-embeddings`. CPU only.
#
# **What changed in this revision:** the metadata file for an array used to be found with a *substring*
# test (`if name in path`), which can bind `Mule.npy` to `Mule_APIkit.csv` — the shape assertion would
# then have failed confusingly, or passed by luck on a same-sized project. The lookup is an exact
# filename match now, and a missing or ambiguous match is reported by name.

# %%
import glob
import os

import numpy as np
import pandas as pd

embs = sorted(glob.glob("/kaggle/input/**/emb/*.npy", recursive=True))
metas = sorted(glob.glob("/kaggle/input/**/meta/*.csv", recursive=True))
print("npy files:", len(embs), "| meta files:", len(metas))
assert len(embs) == 39, f"expected 39 embedding arrays, found {len(embs)}"

meta_by_name = {os.path.basename(p)[:-4]: p for p in metas}
assert len(meta_by_name) == len(metas), "duplicate meta file names"

rows = []
total = 0
for f in embs:
    name = os.path.basename(f)[:-4]
    assert name in meta_by_name, f"{name}.npy has no matching {name}.csv (available: {list(meta_by_name)[:5]}...)"
    a = np.load(f, mmap_mode="r")
    m = pd.read_csv(meta_by_name[name])
    assert a.shape == (len(m), 128, 768), (name, a.shape, len(m))
    assert np.isfinite(np.asarray(a[:4])).all(), f"{name}: non-finite embeddings"
    assert (m.issue_id.astype(str) != "").all(), f"{name}: missing issue_id"
    assert not m.issue_id.duplicated().any(), f"{name}: duplicate issue_id"
    total += os.path.getsize(f)
    rows.append(dict(project=name, n=a.shape[0], size_mb=round(os.path.getsize(f) / 1e6, 1),
                     median_tokens=float(m.n_tokens.median()),
                     pct_full_128=round(float((m.n_tokens >= 128).mean() * 100), 1)))

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/embeddings_check.csv", index=False)
print(df.to_string(index=False))
print(f"TOTAL {total / 1e9:.2f} GB | {len(df)} projects | shapes, finiteness and issue_id keys verified")

# %% [markdown]
# `pct_full_128` counts rows that filled the 128-token window. It is an upper bound on truncation, not
# the truncation rate — the real one, measured before truncation, is in `03`'s `extraction_summary.csv`
# (29.5% of issues at 128 tokens).
