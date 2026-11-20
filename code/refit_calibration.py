#!/usr/bin/env python3
"""Recompute the calibration table on the current model.

Table 14 still held NLL, ECE and accuracy from an early run whose accuracy was 0.671,
while the model now reports 0.83. It also sat under prose claiming that every downstream
number was computed at a fitted temperature, which is not true: all reported open-set
results use the raw scores at T=1. This refits the temperature on the validation knowns
of the five S1 folds, reports before and after, and the prose is corrected to match.

Temperature is fitted by grid search over a multiplicative grid, no scipy required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/tables/calibration_refit.txt")
GRID = np.exp(np.linspace(np.log(0.5), np.log(4.0), 200))


def nll(logits, y, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(y)), y].mean())


def ece(logits, y, T, bins=15):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def auroc(known, unknown):
    s = np.concatenate([known, unknown])
    lab = np.concatenate([np.ones(len(known)), np.zeros(len(unknown))])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/tables/calibration_refit.txt")
    args, _ = ap.parse_known_args()
    global OUT
    OUT = Path(args.out)
    split = json.loads(Path("splits/S1_seed0.json").read_text(encoding="utf-8"))
    rows = []
    for f in range(5):
        run = Path(args.runs) / f"S1_seed0_f{f}"
        if not run.exists():
            continue
        lg = np.load(run / "logits.npy")
        meta = pd.read_csv(run / "meta.csv")
        cls = json.loads((run / "classes.json").read_text(encoding="utf-8"))
        c2i = {c: i for i, c in enumerate(cls)}
        y = np.array([c2i.get(s, -1) for s in meta["species"]])
        val = set(split["folds"][f]["val"])
        unk = set(split["unknown_treeIDs"])
        is_u = meta.treeID.isin(unk).to_numpy()
        is_v = meta.treeID.isin(val).to_numpy() & ~is_u & (y >= 0)
        lv, yv = lg[is_v], y[is_v]

        Ts = min(GRID, key=lambda t: nll(lv, yv, t))
        acc = float((lv.argmax(1) == yv).mean())
        def msp(t):
            z = lv / t; z = z - z.max(1, keepdims=True)
            p = np.exp(z); return p.max(1)
        a1 = auroc(msp(1.0), msp(1.0)[:0]) if False else None
        # AUROC needs unknowns; recompute properly
        z1 = lg / 1.0; z1 = z1 - z1.max(1, keepdims=True); p1 = np.exp(z1); p1 /= p1.sum(1, keepdims=True)
        zT = lg / Ts; zT = zT - zT.max(1, keepdims=True); pT = np.exp(zT); pT /= pT.sum(1, keepdims=True)
        au1 = auroc(p1[is_v].max(1), p1[is_u].max(1))
        auT = auroc(pT[is_v].max(1), pT[is_u].max(1))
        rows.append({
            "fold": f, "T": Ts,
            "nll_before": nll(lv, yv, 1.0), "nll_after": nll(lv, yv, Ts),
            "ece_before": ece(lv, yv, 1.0), "ece_after": ece(lv, yv, Ts),
            "acc": acc,
            "conf_before": float(p1[is_v].max(1).mean()),
            "conf_after": float(pT[is_v].max(1).mean()),
            "auroc_before": au1, "auroc_after": auT,
        })
    d = pd.DataFrame(rows)
    m = d.mean(numeric_only=True)

    lines = ["=" * 62, "CALIBRATION, refitted on the current model (S1, 5 folds)", "=" * 62,
             f"{'quantity':<22}{'T = 1':>12}{'fitted T':>12}"]
    lines.append(f"{'T':<22}{1.0:>12.2f}{m['T']:>12.2f}")
    lines.append(f"{'NLL':<22}{m['nll_before']:>12.4f}{m['nll_after']:>12.4f}")
    lines.append(f"{'ECE':<22}{m['ece_before']:>12.4f}{m['ece_after']:>12.4f}")
    lines.append(f"{'accuracy':<22}{m['acc']:>12.3f}{m['acc']:>12.3f}")
    lines.append(f"{'mean confidence':<22}{m['conf_before']:>12.3f}{m['conf_after']:>12.3f}")
    lines.append(f"{'AUROC (MSP)':<22}{m['auroc_before']:>12.4f}{m['auroc_after']:>12.4f}")
    lines.append("")
    lines.append("per-fold fitted T: " + ", ".join(f"{t:.2f}" for t in d["T"]))
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
