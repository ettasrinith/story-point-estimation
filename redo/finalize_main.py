"""Finalize the PS-3 redo once `madmaxop123/ps3-256-main` has completed on Kaggle.

Run this AFTER the user confirms the Kaggle GPU kernel `ps3-256-main` is COMPLETE:
    python redo/finalize_main.py

It:
  1. Downloads the kernel's output artefacts into results/256_sp100/main/
  2. Re-runs redo/build_comparison.py, which regenerates the comparison
     package (tables, plots, dashboard, README/REPORT injection) including the
     now-available BERT+CNN 256-token results.
  3. Re-renders REPORT.pdf from the refreshed REPORT.md.
  4. Prints a short check that the headline files landed.

All training stays on Kaggle; nothing here trains a model.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "results" / "256_sp100" / "main"
KERNEL = "madmaxop123/ps3-256-main"
PY = r"C:/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe"


def run(cmd):
    print("\n+ " + " ".join(str(c) for c in cmd))
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"FAILED (exit {rc}): {' '.join(str(c) for c in cmd)}")


def main() -> int:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. pull the finished kernel output
    run(["kaggle", "kernels", "output", KERNEL, "-p", str(MAIN_DIR)])
    run(["kaggle", "kernels", "files", KERNEL])

    # 2. regenerate the comparison package with the new data present
    run([PY, str(ROOT / "redo" / "build_comparison.py")])

    # 3. rebuild the final PDF so it matches the refreshed REPORT.md
    run([PY, str(ROOT / "redo" / "build_report_pdf.py")])

    # 4. sanity check the headline artefacts exist
    need = ["master_metrics_256.json", "results_pairs_256.csv",
            "predictions_all_256.csv", "evaluation_summary_256.txt"]
    print("\n=== headline artefact check ===")
    ok = True
    for name in need:
        hits = sorted(MAIN_DIR.rglob(name))
        if hits:
            print(f"  OK   {name}  -> {hits[0]}")
        else:
            ok = False
            print(f"  MISS {name}")
    print("\nDONE." if ok else "\nDONE WITH WARNINGS (some expected files missing).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
