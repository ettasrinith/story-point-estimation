"""Build root-level .ipynb notebooks from the readable percent-format sources in src/.

The .py files in src/ are the source of truth for humans; the .ipynb files at the repo
root are what Kaggle and the GitHub notebook renderer consume.

    python tools/build_notebooks.py            # rebuild every notebook
    python tools/build_notebooks.py 05         # rebuild one (prefix match)

Percent format understood here:
    # %% [markdown]   -> markdown cell
    # %%              -> code cell (everything until the next marker)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

KERNEL_SPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}


def parse(text):
    """Split a percent-format file into (kind, source) cells."""
    lines = text.splitlines()
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        source = "\n".join(buf).strip("\n")
        if source.strip():
            cells.append((kind, source))

    for line in lines:
        marker = re.match(r"^# %%\s*(\[(markdown|md)\])?\s*$", line)
        if marker:
            flush()
            kind = "markdown" if marker.group(2) else "code"
            buf = []
            continue
        if kind is not None:
            buf.append(line)
    flush()
    return cells


def to_lines(source):
    """nbformat convention: every line except the last keeps its newline."""
    parts = source.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build(prefix=None):
    made = []
    for py in sorted(SRC.glob("*.py")):
        if prefix and not py.stem.startswith(prefix):
            continue
        cells = []
        for i, (kind, source) in enumerate(parse(py.read_text(encoding="utf-8"))):
            cell = {
                "cell_type": kind,
                "metadata": {},
                "source": to_lines(source),
                "id": f"{py.stem[:2]}-{i:02d}",
            }
            if kind == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            cells.append(cell)
        nb = {
            "cells": cells,
            "metadata": {
                "kernelspec": KERNEL_SPEC,
                "language_info": {"name": "python", "file_extension": ".py"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        out = ROOT / f"{py.stem}.ipynb"
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        made.append(out)
    return made


if __name__ == "__main__":
    for path in build(sys.argv[1] if len(sys.argv) > 1 else None):
        print(f"built {path.name}")
