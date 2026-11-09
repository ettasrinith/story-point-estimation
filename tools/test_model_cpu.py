"""CPU-only checks on the model code that actually lives inside the notebooks.

The training notebooks need a Kaggle GPU and 12 GB of embeddings, but the correctness of the CNN
masking, the regression head and the fold bookkeeping does not. This pulls the model cells straight
out of 04/05/06 and tests those, so the cells can be checked without a GPU.

    python tools/test_model_cpu.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
L, D = 128, 768


def cell_source(notebook, needle):
    """Source of the first code cell in `notebook` that contains `needle`."""
    nb = json.loads((ROOT / notebook).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and needle in "".join(cell["source"]):
            return "".join(cell["source"])
    raise AssertionError(f"no code cell containing {needle!r} in {notebook}")


def load_namespace(notebook, needle, extra=None):
    """Exec a notebook's model cell in a CPU namespace, so the object under test is the notebook's."""
    ns = dict(np=np, torch=torch, nn=nn, F=F, MAX_LEN=L, EMB_WIDTH=D, BS=8, LR=1e-2,
              MAX_EPOCHS=6, PATIENCE=2, DEV="cpu", re=re)
    ns.update(extra or {})
    exec(compile(cell_source(notebook, needle), f"{notebook}:{needle}", "exec"), ns)
    return ns


def padded_batch(ntok, garbage_scale, seed, real_seed=0):
    """Sequences of different real lengths with loud garbage in their padding.

    Two calls with the same `real_seed` differ ONLY in padded positions.
    """
    ntok = np.asarray(ntok, dtype=np.int64)
    x = np.random.default_rng(real_seed).normal(size=(len(ntok), L, D)).astype(np.float32)
    g = np.random.default_rng(seed)
    for i, k in enumerate(ntok):
        if k < L:
            x[i, k:] = g.normal(size=(L - k, D)).astype(np.float32) * garbage_scale
    return x


class LegacyRegressor(nn.Module):
    """The previous implementation - masked once, after all three convolutions - as the control."""

    def __init__(self, fixed_model):
        super().__init__()
        c = fixed_model
        self.conv = nn.Sequential(
            nn.Conv1d(D, 256, 3, padding=1), nn.ReLU(), nn.Dropout(0.0),
            nn.Conv1d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        with torch.no_grad():            # same weights, so only the masking differs
            self.conv[0].load_state_dict(c.conv1.state_dict())
            self.conv[3].load_state_dict(c.conv2.state_dict())
            self.conv[5].load_state_dict(c.conv3.state_dict())
            self.head[0].load_state_dict(c.fc1.state_dict())
            self.head[2].load_state_dict(c.fc2.state_dict())

    def forward(self, x, ntok):
        h = self.conv(x.transpose(1, 2))
        m = (torch.arange(x.size(1), device=x.device)[None, :] < ntok[:, None]).unsqueeze(1)
        return self.head(h.masked_fill(~m, float("-inf")).max(dim=2).values).squeeze(-1)


def test_padding_is_inert(ns, model_name="CNNRegressor"):
    Model = ns[model_name]
    torch.manual_seed(0)
    model = Model().eval()
    ntok = np.array([7, 40, 128])
    nt = torch.tensor(np.clip(ntok, 1, L))
    a, b = padded_batch(ntok, 50.0, seed=1), padded_batch(ntok, 100.0, seed=2)
    assert np.array_equal(a[:, :7], b[:, :7]), "test data is wrong: the real tokens differ"
    with torch.no_grad():
        delta = float((model(torch.tensor(a), nt) - model(torch.tensor(b), nt)).abs().max())
    assert np.isfinite(delta), "model returned a non-finite output"
    assert delta < 1e-5, f"padded positions still leak into the output (max delta {delta:.3e})"

    legacy = LegacyRegressor(model).eval()
    with torch.no_grad():
        control = float((legacy(torch.tensor(a), nt) - legacy(torch.tensor(b), nt)).abs().max())
    assert control > 1e-3, f"the control model did not leak ({control:.3e}) - this test proves nothing"
    print(f"  padding inert: fixed {delta:.1e} vs old mask-after-conv {control:.2f} "
          "(same weights, pads-only difference in the input)")


def test_output_non_negative(ns, model_name="CNNRegressor"):
    model = ns[model_name]()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ntok = np.full(16, 30)
    x = torch.tensor(padded_batch(ntok, 5.0, seed=4))
    nt = torch.tensor(np.clip(ntok, 1, L))
    y = torch.rand(16, generator=torch.Generator().manual_seed(3)) * 8 + 1
    for _ in range(60):
        loss = F.l1_loss(model(x, nt), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(x, nt)
    assert (pred >= 0).all(), f"negative story point predicted: {pred.min():.4f}"
    print(f"  non-negative head: min of {len(pred)} fitted predictions = {float(pred.min()):.3f}")


def test_fold_reports_best_epoch(ns):
    """A fold reports its best validation MAE and that best epoch, not the stopping epoch - and the
    weights handed back are the ones that scored that value."""
    ntok = np.full(24, 50)
    x = padded_batch(ntok, 2.0, seed=5)
    y = (np.arange(24) % 5 + 1).astype(np.float32)
    nt = np.clip(ntok, 1, L)
    model, hist, best_val, best_epoch = ns["train_model"](
        x[:16], y[:16], nt[:16], epochs=5, bs=8, seed=0, Xva=x[16:], yva=y[16:], nva=nt[16:])
    assert hist and best_val is not None, "no validation result returned"
    assert abs(best_val - min(hist)) < 1e-9, f"best_val {best_val} != min(hist) {min(hist)}"
    assert best_epoch == hist.index(best_val) + 1, f"best_epoch {best_epoch} != argmin + 1"
    assert len(hist) >= best_epoch, "best epoch beyond the number of epochs run"
    live = float(np.abs(ns["predict"](model, x[16:], nt[16:]) - y[16:]).mean())
    assert abs(live - best_val) < 5e-3, f"restored weights give {live}, reported best was {best_val}"
    print(f"  fold bookkeeping: hist={['%.3f' % v for v in hist]} -> reports {best_val:.3f} "
          f"(epoch {best_epoch} of {len(hist)} run); restored weights reproduce it")

    noval = ns["train_model"](x[:8], y[:8], nt[:8], epochs=2, bs=4, seed=0)
    assert len(noval) == 4 and noval[2] is None and noval[3] is None
    print("  no-validation mode returns (model, [], None, None) as documented")


def test_pooled_fold(ns):
    idx = np.arange(20)
    model, hist, best_val, best_epoch = ns["train_one"](idx[:12], epochs=4, bs=6, seed=0, va_idx=idx[12:])
    assert abs(best_val - min(hist)) < 1e-9 and best_epoch == hist.index(best_val) + 1
    pred = ns["predict_raw"](model, idx[12:])
    assert (pred >= 0).all() and np.isfinite(pred).all(), "expm1 back-transform broken"
    print(f"  pooled fold: reports {best_val:.3f} points at epoch {best_epoch} of {len(hist)} run; "
          f"log-space target back to [{pred.min():.2f}, {pred.max():.2f}] points")


def test_split_cell_matches_the_baseline_rows():
    """Run notebook 05's own split cell on a synthetic table and check it picks the rows 02 picks.

    05 stores splits as per-project ranks because the embedding files are indexed that way, and a
    rank is only sound if converting the draw keeps pointing at the rows it started from. Below
    pandas 3, `Index.to_numpy()` returns the Index's own buffer, so `rng.shuffle(idx)` reordered
    `g.index` as well and the ranks were read from the shuffled order - the real Kaggle run then
    trained on a different partition than the baseline it was compared with (05's median check is
    what surfaced it).
    """
    import tempfile
    import pandas as pd

    tmp = Path(tempfile.mkdtemp())
    lab = np.random.default_rng(0)
    sizes = {"alpha": 37, "beta project ": 61, "gamma": 24}
    rows = []
    for proj, n in sizes.items():
        for i in range(n):
            rows.append(dict(project=proj, issue_id=f"{proj}-{i}", issue_key=f"K{i}",
                             text="some issue text", storypoint=float(lab.integers(1, 9))))
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp / "all_issues_sp.csv", index=False)

    # 02's partition, computed the way 02 states it: shuffle the labels, split by label.
    ref = frame.copy()
    ref["storypoint"] = pd.to_numeric(ref["storypoint"], errors="coerce")
    ref = ref.dropna(subset=["storypoint"])
    ref = ref[(ref.storypoint > 0) & (ref.storypoint <= 40)]
    ref["project"] = ref["project"].astype(str).str.strip()
    rng = np.random.default_rng(42)
    want, base_rows = {}, []
    for proj, g in ref.groupby("project"):
        idx = np.array(g.index)
        rng.shuffle(idx)
        n_test = max(1, int(round(0.2 * len(idx))))
        want[proj] = (set(idx[:n_test].tolist()), set(idx[n_test:].tolist()))
        base_rows.append(dict(source=proj, target=proj + " [WITHIN]",
                              pred=float(ref.loc[idx[n_test:], "storypoint"].median()),
                              n_train=len(idx) - n_test, n_test=n_test))
    pd.DataFrame(base_rows).to_csv(tmp / "baseline_results.csv", index=False)

    class FakeGlob:
        @staticmethod
        def glob(pattern, recursive=False):
            return [str(tmp / "all_issues_sp.csv")] if "all_issues_sp" in pattern \
                else [str(tmp / "baseline_results.csv")]

    real_glob = sys.modules.get("glob")
    sys.modules["glob"] = FakeGlob          # the cell imports glob itself, so swap it at the source
    try:
        ns = load_namespace("05_run_all_pairs.ipynb", "splits = {}", extra={"SEED": 42})
    finally:
        if real_glob is not None:
            sys.modules["glob"] = real_glob
        else:
            del sys.modules["glob"]
    for proj, (te, tr) in want.items():
        g = ref[ref.project == proj]
        got_te = set(g.index.values[ns["splits"][proj]["test"]].tolist())
        got_tr = set(g.index.values[ns["splits"][proj]["train"]].tolist())
        assert got_te == te and got_tr == tr, f"{proj}: 05 selected a different partition than 02"
        assert len(got_te) + len(got_tr) == len(g)
    print(f"  splits: 05's cell reproduces 02's rows for {len(want)} projects "
          "(ranks, shuffled draws and label order agree)")


def main():
    print("05_run_all_pairs.ipynb")
    ns5 = load_namespace("05_run_all_pairs.ipynb", "def train_model(")
    test_padding_is_inert(ns5)
    test_output_non_negative(ns5)
    test_fold_reports_best_epoch(ns5)
    test_split_cell_matches_the_baseline_rows()

    print("04_cnn_model.ipynb")
    ns4 = load_namespace("04_cnn_model.ipynb", "def train_model(")
    test_padding_is_inert(ns4)
    test_output_non_negative(ns4)

    print("06_pooled_training.ipynb")
    ns6 = load_namespace("06_pooled_training.ipynb", "def train_one(",
                         extra={"BIG": padded_batch(np.full(20, 60), 2.0, seed=6).astype(np.float16),
                                "NTK": np.full(20, 60, dtype=np.int32),
                                "Y": (np.arange(20) % 6 + 1).astype(np.float32)})
    ns6["YLOG"] = np.log1p(ns6["Y"])
    test_padding_is_inert(ns6)
    test_pooled_fold(ns6)

    print("\nall CPU checks passed")


if __name__ == "__main__":
    sys.exit(main())
