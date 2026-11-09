# 03_bert_embeddings.py
#
# One-time GPU job. The arrays it writes are what 04/05/06 read; regenerating them is not needed for
# any change in those notebooks.

# %% [markdown]
# # 03 — BERT Token Embedding Extraction
#
# Runs `bert-base-uncased` (frozen, inference only) over every issue text once and stores the
# token-level last-hidden-state vectors per project, so the CNN notebooks never touch BERT again.
#
# **Input:** dataset `tawos-extracted` (`all_issues_sp.csv` from `01`).
# **Output:**
#
# | File | Contents |
# |---|---|
# | `emb/<PROJECT>.npy` | float16 `[n_issues, 128, 768]`, one row per issue, same order as its meta CSV |
# | `meta/<PROJECT>.csv` | `issue_id, issue_key, storypoint, n_tokens` |
# | `extraction_summary.csv` | per project: issue count, size, true truncation rate at 128 and 256 |
# | `run_metadata.json` | model identifier and revision, library versions, settings |
#
# Total 12.2 GB. Settings: `max_length=128`, batch 64, fp16 autocast. Same label filter as `02`
# (story points in (0, 40]). Needs GPU and Internet ON (weights download).
#
# **What changed in this revision:** the truncation rate is measured properly. `n_tokens` is counted
# *after* truncation, so `n_tokens >= 128` only says a sequence filled the window — it is not the
# share of issues that lost content. The length is now taken from an untruncated tokenization pass
# first. Fixing the measurement barely moved the number (29.8% → 29.5%), but the earlier figure of
# "about 13–15%" quoted in the README and report matched neither and has been corrected.

# %%
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import transformers

MAX_LEN = 128
BS = 64
MODEL = "bert-base-uncased"

hits = glob.glob("/kaggle/input/**/all_issues_sp.csv", recursive=True)
assert hits, "attach the tawos-extracted dataset as Input"
df = pd.read_csv(hits[0])
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
df = df[(df.storypoint > 0) & (df.storypoint <= 40)]
df["text"] = df["text"].fillna("").astype(str)
df = df.reset_index(drop=True)
print(len(df), "issues |", df.project.nunique(), "projects")

# %%
import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).cuda().eval()
torch.set_grad_enabled(False)

revision = None
try:                                  # needs Internet; a plain identifier is enough if offline
    from huggingface_hub import model_info
    revision = model_info(MODEL).sha
except Exception:
    pass

os.makedirs("/kaggle/working", exist_ok=True)
json.dump({
    "model": MODEL,
    "revision": revision,
    "max_length": MAX_LEN,
    "batch_size": BS,
    "dtype": "float16 autocast, fp16 storage",
    "issues": int(len(df)),
    "projects": int(df.project.nunique()),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "numpy": np.__version__,
    "cuda_device": torch.cuda.get_device_name(0),
}, open("/kaggle/working/run_metadata.json", "w"), indent=2)
print("gpu:", torch.cuda.get_device_name(0),
      "| params:", sum(p.numel() for p in model.parameters()) // 1_000_000, "M",
      "| revision:", revision)

# %% [markdown]
# ## Extraction loop
#
# Two passes per batch: one without truncation to get the true token length (for the truncation
# statistics at 128 and 256) and one padded to exactly `MAX_LEN` for the forward pass. `n_tokens`
# records the real, untruncated length so downstream code can tell a full window from a short issue.

# %%
os.makedirs("/kaggle/working/emb", exist_ok=True)
os.makedirs("/kaggle/working/meta", exist_ok=True)


def safe(n):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(n))


projects = sorted(df.project.unique())
summary = []
t0 = time.time()
for pi, proj in enumerate(projects):
    g = df[df.project == proj].reset_index(drop=True)
    out = np.empty((len(g), MAX_LEN, 768), dtype=np.float16)
    real_len = np.empty(len(g), dtype=np.int32)
    ntok = np.empty(len(g), dtype=np.int32)
    for b0 in range(0, len(g), BS):
        batch = list(g.text.values[b0:b0 + BS])
        # No truncation: the true length, before anything is cut.
        real_len[b0:b0 + len(batch)] = [len(ids) for ids in
                                        tok(batch, add_special_tokens=True, truncation=False)["input_ids"]]
        enc = tok(batch, padding="max_length", truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            hs = model(**enc).last_hidden_state
        out[b0:b0 + len(batch)] = hs.half().cpu().numpy()
        ntok[b0:b0 + len(batch)] = enc["attention_mask"].sum(1).cpu().numpy()
        del enc, hs

    name = safe(proj)
    np.save(f"/kaggle/working/emb/{name}.npy", out)
    meta = g[["issue_id", "issue_key", "storypoint"]].copy()
    meta["n_tokens"] = real_len
    meta.to_csv(f"/kaggle/working/meta/{name}.csv", index=False)
    summary.append(dict(
        project=proj, file=f"{name}.npy", n=len(g), size_mb=round(out.nbytes / 1e6, 1),
        tokens_median=float(np.median(real_len)),
        trunc_pct_128=round(float((real_len > 128).mean() * 100), 1),
        trunc_pct_256=round(float((real_len > 256).mean() * 100), 1),
        content_lost_at_128_pct=round(float(np.maximum(real_len - 128, 0).sum() / real_len.sum() * 100), 1),
    ))
    del out, ntok, real_len
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{pi + 1}/{len(projects)}] {proj}: {len(g)} issues "
          f"| truncated at 128: {summary[-1]['trunc_pct_128']}% | {time.time() - t0:.0f}s", flush=True)

sm = pd.DataFrame(summary)
sm.to_csv("/kaggle/working/extraction_summary.csv", index=False)
overall = dict(
    issues=int(sm.n.sum()),
    trunc_pct_128=round(float(sm.trunc_pct_128.mean()), 1),
    trunc_pct_256=round(float(sm.trunc_pct_256.mean()), 1),
    content_lost_at_128_pct=round(float(sm.content_lost_at_128_pct.mean()), 1),
)
print("\nOVERALL", overall, "| DONE in", round((time.time() - t0) / 60, 1), "min")

# %% [markdown]
# ## Verification
#
# Every file is checked for shape, row count against the metadata and finite values before the 12 GB
# is published. `issue_id` is what later notebooks key on when they confirm that row *i* of an array
# really is issue *i* of the filtered frame.

# %%
total = 0
for _, r in pd.read_csv("/kaggle/working/extraction_summary.csv").iterrows():
    a = np.load(f"/kaggle/working/emb/{r.file}", mmap_mode="r")
    m = pd.read_csv(f"/kaggle/working/meta/{r.file[:-4]}.csv")
    assert a.shape == (len(m), MAX_LEN, 768), (r.file, a.shape, len(m))
    assert len(m) == r.n, r.file
    assert np.isfinite(np.asarray(a[:4])).all(), r.file
    total += a.nbytes
print(f"all {len(sm)} project files verified | total {total / 1e9:.2f} GB")
print("\nmost-truncated projects (real lengths, before truncation):")
print(sm.nlargest(6, "trunc_pct_128")[["project", "n", "tokens_median",
                                        "trunc_pct_128", "trunc_pct_256"]].to_string(index=False))

# %% [markdown]
# ## Notes
#
# - float16 halves the file size with no practical loss for a CNN input layer.
# - Row order inside each `.npy` is the row order of `meta/<PROJECT>.csv`, which is the order the
#   issue appears in the filtered `all_issues_sp.csv` frame. `05`/`06` assert this by `issue_id`
#   before training rather than trusting it.
# - `n_tokens` is the **untruncated** length, so it can exceed 128; downstream clamps it and uses it
#   as the padding mask for the convolution and the max pool.
# - At 128 tokens roughly 29% of issues lose text and about 42% of token content is cut. The
#   Atlassian projects are the worst (Jira Cloud/Server ~92% truncated), which is visible in their
#   pair results. Raising `MAX_LEN` to 256 would cut that to ~9% but doubles the storage and the CNN
#   sequence length — left as future work, since the committed embeddings are at 128.
# - File names come from the raw TAWOS project names, three of which end in a space
#   (`The_Titanium_SDK_.npy`). `05`/`06` resolve those; do not rename the existing arrays.
