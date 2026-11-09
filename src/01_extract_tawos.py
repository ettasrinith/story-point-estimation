# 01_extract_tawos.py
#
# Extraction and cleaning of the official TAWOS dump. Everything downstream (02 baseline,
# 03 embeddings, 04-06 models) reads the all_issues_sp.csv this notebook produces.

# %% [markdown]
# # 01 — TAWOS Data Extraction
#
# Loads the official TAWOS MySQL dump (Tawosi et al., MSR 2022, DOI 10.5522/04/21308124) into a MySQL
# server started inside the Kaggle notebook, joins `Issue` to `Project`, and exports one CSV per
# project plus a single `all_issues_sp.csv` of every issue that carries a story point.
#
# **Input:** Kaggle dataset containing `TAWOS.sql` (or `TAWOS.sql.zip`).
# **Output (in `/kaggle/working`):**
#
# | File | Contents |
# |---|---|
# | `csv/<PROJECT>.csv` | one per project: `project, issue_id, issue_key, summary, description, created, storypoint, text` |
# | `all_issues_sp.csv` | all projects concatenated |
# | `manifest.csv` | per-project counts and story point statistics |
# | `filter_impact.csv` | what the experimental `(0, 40]` label filter removes, per project |
# | `eda_distributions.png` | distributions *after* the filter, plus a before/after panel |
# | `tawos_clean_csv.zip` | the per-project CSVs zipped |
#
# Settings: CPU only, Internet ON (apt install of MySQL).
#
# **What changed in this revision:** the story-point column is now taken from the `Issue` table when
# `Issue` has one instead of from whichever table happens to sort first; the post-estimation change
# flags are carried through so records edited after being estimated can be counted; empty-text issues
# are dropped explicitly; the label filter is reported per project instead of asserted in prose; and
# the EDA plots are drawn after the filter so extreme values no longer squash the axis.

# %%
import subprocess
import time


def run(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


# The dump comes from MySQL 8; its utf8mb4_0900_ai_ci collation is unknown to MariaDB, so MySQL is
# preferred and MariaDB is only a fallback.
run("apt-get update -qq")
r = run("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mysql-server >/tmp/apt.log 2>&1",
        capture_output=True, text=True)
ENGINE = "mysql" if r.returncode == 0 else "mariadb"
if ENGINE == "mariadb":
    print("mysql-server install failed -> mariadb fallback")
    run("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >>/tmp/apt.log 2>&1")

cnf = ("[mysqld]\nmax_allowed_packet=512M\ninnodb_flush_log_at_trx_commit=2\n"
       "innodb_buffer_pool_size=8G\nskip-log-bin\n")
open("/etc/mysql/conf.d/load.cnf", "w").write(cnf)
run(f"service {ENGINE} start")
time.sleep(8)
if run("mysql -e 'SELECT VERSION()'", capture_output=True, text=True).returncode != 0:
    print("service start failed, trying mysqld_safe")
    subprocess.Popen("mysqld_safe >/tmp/mysqld.log 2>&1 &", shell=True)
    time.sleep(20)

# pandas connects over TCP, and root is socket-authenticated on both engines, so a TCP user is created.
run("mysql -e " + repr("CREATE USER IF NOT EXISTS 'k'@'127.0.0.1' IDENTIFIED BY 'k'; "
                       "GRANT ALL PRIVILEGES ON *.* TO 'k'@'127.0.0.1';"))
r = run("mysql -uk -pk -h 127.0.0.1 -e 'SELECT VERSION()'", capture_output=True, text=True)
print(r.stdout or r.stderr)
assert r.returncode == 0, "no working MySQL/MariaDB"

# %%
import glob
import os
import zipfile

os.makedirs("/kaggle/tmp", exist_ok=True)
cands = glob.glob("/kaggle/input/**/*.sql*", recursive=True)
print("found in inputs:", cands)
for c in cands:
    if c.endswith(".zip"):
        with zipfile.ZipFile(c) as z:
            z.extractall("/kaggle/tmp")
sqls = glob.glob("/kaggle/tmp/**/*.sql", recursive=True) + [c for c in cands if c.endswith(".sql")]
assert sqls, "No .sql file found - is the TAWOS dataset attached?"
DUMP = max(sqls, key=os.path.getsize)
print("DUMP =", DUMP, "|", round(os.path.getsize(DUMP) / 1e6, 1), "MB")

# %%
run("mysql -e 'DROP DATABASE IF EXISTS tawos; CREATE DATABASE tawos'")
DUMPX = DUMP
if ENGINE == "mariadb":
    # Rewrite the collation MariaDB lacks. The input mount is read-only, so work on a copy.
    import shutil
    DUMPX = "/kaggle/tmp/TAWOS_fixed.sql"
    if not os.path.exists(DUMPX):
        shutil.copy(DUMP, DUMPX + ".tmp")
        run(f"sed -i 's/utf8mb4_0900_ai_ci/utf8mb4_general_ci/g' '{DUMPX}.tmp'")
        os.rename(DUMPX + ".tmp", DUMPX)
t0 = time.time()
r = subprocess.run(f"mysql --max_allowed_packet=512M tawos < '{DUMPX}'", shell=True,
                   capture_output=True, text=True)
print("load rc:", r.returncode, "| minutes:", round((time.time() - t0) / 60, 1))
if r.returncode != 0:
    print(r.stderr[-3000:])
r = run("mysql tawos -e 'SHOW TABLES'", capture_output=True, text=True)
print(r.stdout)
assert r.stdout.strip(), "DB empty after load"

# %% [markdown]
# ## Inspect the schema before trusting it
#
# The dump's table and column names have changed between TAWOS releases, so the notebook discovers
# them rather than hard-coding them. The printed table is what to check by eye: the story point must
# come from a column that really is the issue's estimate, not from a coincidence of table ordering.

# %%
import pandas as pd
import sqlalchemy

subprocess.run("pip install -q pymysql", shell=True)
engine = sqlalchemy.create_engine("mysql+pymysql://k:k@127.0.0.1/tawos")

alltabs = pd.read_sql(
    "SELECT TABLE_NAME AS tname, IFNULL(TABLE_ROWS,0) AS nrows "
    "FROM information_schema.tables WHERE table_schema='tawos' ORDER BY nrows DESC", engine)
print(alltabs.to_string(index=False))


def cols(table):
    d = pd.read_sql(
        "SELECT COLUMN_NAME AS cname, DATA_TYPE AS dtype FROM information_schema.columns "
        "WHERE table_schema='tawos' AND LOWER(TABLE_NAME)=LOWER(%s)", engine, params=[(table,)])
    d.columns = ["column_name", "data_type"]
    return d.column_name.tolist()


def real_name(lower):
    r = pd.read_sql("SELECT TABLE_NAME AS tname FROM information_schema.tables "
                    "WHERE table_schema='tawos' AND LOWER(TABLE_NAME)=%s", engine, params=[(lower,)])
    return r.tname.iloc[0] if len(r) else None


ISSUE, PROJ = real_name("issue"), real_name("project")
print("ISSUE =", ISSUE, "| PROJ =", PROJ)
print("Issue columns :", list(cols(ISSUE)))
print("Project columns:", list(cols(PROJ)))
sp_report = []
for t in alltabs.tname:
    hits = [c for c in cols(t) if "story" in str(c).lower()]
    if hits:
        sp_report.append((t, hits))
print("tables carrying a 'story' column:", sp_report)

# %% [markdown]
# ## Build the extraction query
#
# The story point is read from the `Issue` table whenever `Issue` has one. Only if it does not does
# the notebook join the first other table that carries a story column — the earlier version joined
# whichever such table appeared first in row-count order, which is not a guarantee.
#
# The post-estimation change flags are selected when the schema provides them. An issue whose
# description was rewritten after the team estimated it hands the model information nobody had at
# estimation time, so those rows are counted and reported, not silently dropped.

# %%
def pick(clist, *pats, required=True):
    for pat in pats:
        for n in clist:
            if str(n).lower() == pat:
                return n
    for pat in pats:
        for n in clist:
            if pat in str(n).lower():
                return n
    if required:
        raise KeyError(f"no column matching {pats} in {list(clist)}")
    return None


ic, pc = cols(ISSUE), cols(PROJ)
summary_c = pick(ic, "title", "summary")
desc_c = pick(ic, "description_text", "description", required=False)
proj_fk = pick(ic, "project")
proj_name = pick(pc, "name", "title")
created_c = pick(ic, "creation_date", "created", "created_at", required=False)
key_c = pick(ic, "issue_key", "key", "jira_id", required=False)

# Optional audit columns: when text or the estimate was touched after estimation.
title_chg_c = pick(ic, "title_changed_after_estimation", "title_changed", required=False)
desc_chg_c = pick(ic, "description_changed_after_estimation", "description_changed", required=False)
sp_chg_c = pick(ic, "story_point_changed_after_estimation", "story_point_changed", required=False)
est_date_c = pick(ic, "estimation_date", "story_point_date", "resolved", required=False)
print("audit columns found:", {"title_changed": title_chg_c, "description_changed": desc_chg_c,
                               "story_point_changed": sp_chg_c, "estimation_date": est_date_c})

sel = "p.{0} AS project, i.ID AS issue_id".format(proj_name)
if key_c:
    sel += f", i.{key_c} AS issue_key"
sel += f", i.{summary_c} AS summary"
if desc_c:
    sel += f", i.{desc_c} AS description"
if created_c:
    sel += f", i.{created_c} AS created"

issue_sp = pick(ic, "story_point", "story", "estimate", required=False)
if issue_sp:
    sel += f", i.{issue_sp} AS storypoint"
    for alias, col in [("title_changed_after_est", title_chg_c),
                       ("description_changed_after_est", desc_chg_c),
                       ("sp_changed_after_est", sp_chg_c),
                       ("estimation_date", est_date_c)]:
        if col:
            sel += f", i.{col} AS {alias}"
    q = f"SELECT {sel} FROM {ISSUE} i JOIN {PROJ} p ON i.{proj_fk}=p.ID"
else:
    other = next((t for t, hits in sp_report if t != ISSUE), None)
    assert other, "no story-point column found on Issue or in any other table"
    oc = cols(other)
    sp_val = pick(oc, "value", "point", "story")
    sp_fk = pick(oc, "issue")
    sel += f", s.{sp_val} AS storypoint"
    q = (f"SELECT {sel} FROM {ISSUE} i JOIN {PROJ} p ON i.{proj_fk}=p.ID "
         f"JOIN {other} s ON s.{sp_fk}=i.ID")

print("QUERY: " + q)
df = pd.read_sql(q, engine)
print("rows:", df.shape)
assert df.storypoint.notna().any(), "story point column is entirely null - wrong column picked"

# %% [markdown]
# ## Clean
#
# Stages are counted as they happen so the numbers in the report are measurements, not descriptions.

# %%
import re

SP_MIN, SP_MAX = 0, 40          # experimental label filter, reported per project below
n_all = len(df)
df["storypoint"] = pd.to_numeric(df["storypoint"], errors="coerce")
df = df.dropna(subset=["storypoint"])
n_has_sp = len(df)
df = df.drop_duplicates("issue_id")
n_unique = len(df)


def clean_text(x):
    if not isinstance(x, str):
        return ""
    x = re.sub(r"<[^>]+>", " ", x)
    return re.sub(r"\s+", " ", x).strip()


for c in ["summary", "description"]:
    if c in df:
        df[c] = df[c].fillna("").map(clean_text)
df["text"] = (df["summary"] + (" " + df["description"] if "description" in df else "")).str.strip()

# An issue with no title and no description gives the model nothing; keep title-only and
# description-only rows, drop the ones with neither.
n_no_text = int((df["text"].str.len() == 0).sum())
df = df[df["text"].str.len() > 0]
n_text = len(df)
df = df.reset_index(drop=True)

print(f"all issues joined to a project : {n_all}")
print(f"after numeric story point      : {n_has_sp}")
print(f"after unique issue_id          : {n_unique}")
print(f"dropped for empty title+body   : {n_no_text}")
print(f"final issues with story points : {n_text}")

for flag in ["title_changed_after_est", "description_changed_after_est", "sp_changed_after_est"]:
    if flag in df:
        pct = 100 * float(pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(bool).mean())
        print(f"{flag}: {pct:.1f}% of issues ({int(pd.to_numeric(df[flag], errors='coerce').fillna(0).astype(bool).sum())})")

# %% [markdown]
# ## Export, manifest and the cost of the label filter
#
# `filter_impact.csv` is the justification for `(0, 40]`: per project, how many rows are 0, how many
# exceed 40, what the maximum really is. The values that motivate the cut are visible here — one
# project's story points reach 26,710,114.

# %%
import shutil

os.makedirs("/kaggle/working/csv", exist_ok=True)
rows = []
for proj, g in df.groupby("project"):
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(proj))
    g.to_csv(f"/kaggle/working/csv/{name}.csv", index=False)
    rows.append(dict(project=proj, file=f"{name}.csv", n=len(g),
                     sp_min=g.storypoint.min(), sp_median=g.storypoint.median(),
                     sp_mean=round(g.storypoint.mean(), 2), sp_max=g.storypoint.max()))
man = pd.DataFrame(rows).sort_values("n", ascending=False)
man.to_csv("/kaggle/working/manifest.csv", index=False)
df.to_csv("/kaggle/working/all_issues_sp.csv", index=False)

kept = df[(df.storypoint > SP_MIN) & (df.storypoint <= SP_MAX)]
impact = pd.DataFrame({
    "extracted": df.groupby("project").size(),
    "sp_zero": df[df.storypoint == SP_MIN].groupby("project").size(),
    "sp_above_40": df[df.storypoint > SP_MAX].groupby("project").size(),
    "kept_for_experiments": kept.groupby("project").size(),
    "text_median_chars": df.groupby("project").text.str.len().median(),
}).fillna(0).astype({"sp_zero": int, "sp_above_40": int, "kept_for_experiments": int})
impact["removed"] = impact.extracted - impact.kept_for_experiments
impact = impact.sort_values("removed", ascending=False)
impact.to_csv("/kaggle/working/filter_impact.csv")
print(man.to_string(index=False))
print(f"\n(0,{SP_MAX}] filter: {len(df)} -> {len(kept)} issues "
      f"({int((df.storypoint == 0).sum())} zeros, {int((df.storypoint > SP_MAX).sum())} above {SP_MAX})")
print("\nper-project impact of the label filter:")
print(impact.head(12).to_string())
shutil.make_archive("/kaggle/working/tawos_clean_csv", "zip", "/kaggle/working/csv")
print("\nDONE. Files in /kaggle/working:", sorted(os.listdir("/kaggle/working")))

# %% [markdown]
# ## EDA
#
# Drawn on the filtered frame. The previous figure plotted raw story points, and a single project
# with values in the tens of thousands stretched the x-axis until every real distribution was a
# spike at the left edge.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(16, 11))

top = kept.groupby("project").size().sort_values(ascending=False).head(15)
ax[0, 0].barh(range(len(top)), top.values, color="steelblue")
ax[0, 0].set_yticks(range(len(top)), top.index, fontsize=8)
ax[0, 0].invert_yaxis()
ax[0, 0].set_xlabel("issues with a story point in (0, 40]")
ax[0, 0].set_title("Usable issues per project (15 largest)")

labels = [p for p, _ in df.groupby("project").size().sort_values(ascending=False).head(8)]
ax[0, 1].boxplot([kept[kept.project == p].storypoint.values for p in labels],
                 tick_labels=[p[:16] for p in labels], showfliers=False)
ax[0, 1].set_ylabel("story point")
ax[0, 1].set_title("Story point spread per project (filtered, outliers hidden)")
ax[0, 1].tick_params(axis="x", rotation=45, labelsize=7)

summary_tbl = pd.DataFrame({
    "median": kept.groupby("project").storypoint.median(),
    "mean": kept.groupby("project").storypoint.mean(),
    "n": kept.groupby("project").size(),
}).sort_values("median")
ax[1, 0].scatter(summary_tbl.n, summary_tbl.median, s=18, alpha=0.8)
ax[1, 0].set_xscale("log")
ax[1, 0].set_xlabel("project size (log)")
ax[1, 0].set_ylabel("median story point")
ax[1, 0].set_title("Projects do not share a scale — the reason cross-project MAE is hard")

counts = kept.storypoint.round().value_counts().sort_index()
ax[1, 1].bar(counts.index, counts.values, color="darkseagreen")
ax[1, 1].set_xlabel("story point")
ax[1, 1].set_ylabel("issues")
ax[1, 1].set_yscale("log")
ax[1, 1].set_title(f"Whole-dataset story point frequencies (n = {len(kept)})")

plt.tight_layout()
plt.savefig("/kaggle/working/eda_distributions.png", dpi=110)
plt.show()

# %% [markdown]
# ## After the run
#
# 1. **Save Version → Save & Run All (commit)**, then publish the output as a Dataset
#    (`tawos-extracted`). Every later notebook attaches it.
# 2. Cross-check `manifest.csv` row counts against the per-project numbers published by Tawosi et al.
# 3. Keep `filter_impact.csv` — the report quotes it for the label-filter justification.
