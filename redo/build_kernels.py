"""Build the two Kaggle kernels used by the PS-3 redo.

Sources live in ``redo/src/*.py`` in percent format (``# %%`` markers). For each kernel this
script writes ``redo/kernels/<slug>/`` containing the .ipynb and the kernel-metadata.json that
``kaggle kernels push -p <folder>`` consumes.

    python redo/build_kernels.py            # build both
    python redo/build_kernels.py 10         # build one (prefix match)

Only two kernels are needed now:

* ``ps3-256-audit`` (CPU) — corpus, median-baseline and truncation measurements on the
  (0,100] corpus. Outputs live in ``results/256_sp100/audit/``.
* ``ps3-256-main``  (GPU) — the assigned 39-pair BERT + CNN experiment at 256 tokens.
  Outputs are downloaded into ``results/256_sp100/main/`` by ``finalize_main.py``.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
OUT = ROOT / "kernels"

OWNER = "madmaxop123"
DATASETS = ["ettasrinithreddy/tawos-extracted"]
EMB_KERNELS = ["madmaxop123/tawos-bert-256-a", "madmaxop123/tawos-bert-256-b"]

# Titles are written so that the Kaggle-generated slug equals the id slug exactly.
CONFIG = {
    "01_audit": dict(slug="ps3-256-audit", title="ps3 256 audit",
                     src="01_audit.py", gpu=False, internet=False),
    "10_main": dict(slug="ps3-256-main", title="ps3 256 main",
                    src="10_main.py", gpu=True, internet=False),
}

KERNEL_SPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


def parse(text):
    lines = text.splitlines()
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        source = "\n".join(buf).strip("\n")
        if source.strip():
            cells.append((kind, source))

    for line in lines:
        m = re.match(r"^# %%\s*(\[(markdown|md)\])?\s*$", line)
        if m:
            flush()
            kind = "markdown" if m.group(2) else "code"
            buf = []
            continue
        if kind is not None:
            buf.append(line)
    flush()
    return cells


def to_lines(source):
    parts = source.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build(stem):
    cfg = CONFIG[stem]
    slug = cfg["slug"]
    cells_src = parse((SRC / cfg["src"]).read_text(encoding="utf-8"))

    cells = []
    for i, (kind, source) in enumerate(cells_src):
        cell = {"cell_type": kind, "metadata": {}, "source": to_lines(source),
                "id": f"{stem[:2]}-{i:02d}"}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {"cells": cells,
          "metadata": {"kernelspec": KERNEL_SPEC,
                       "language_info": {"name": "python", "file_extension": ".py"}},
          "nbformat": 4, "nbformat_minor": 5}

    folder = OUT / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": f"{OWNER}/{slug}",
        "title": cfg["title"],
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": cfg["gpu"],
        "enable_internet": cfg["internet"],
        "dataset_sources": DATASETS,
        "competition_sources": [],
        "kernel_sources": EMB_KERNELS,
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built", folder)


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else ""
    for stem in CONFIG:
        if stem.startswith(prefix):
            build(stem)
