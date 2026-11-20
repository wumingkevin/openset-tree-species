#!/usr/bin/env python3
"""Build the backbone x partition matrix.

The paper argues that the ordering of the partitions is a property of the data rather than of
one architecture. That claim needs the same partition set run through a second backbone, and
this compares the two on every partition: held-out accuracy, AUROC, the acceptance rate and
the coarse-level recall, with the gap between backbones beside each.

    python code/backbone_matrix.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VIEWS = Path("results/all")
PC = Path("results/all_pc")
OUT = Path("results/tables/backbone_matrix.txt")

PARTITIONS = [
    ("S1_seed0", "S1 random species"),
    ("S2_acer", "S2 genus Acer"),
    ("S2_quercus", "S2 genus Quercus"),
    ("S2_pinus", "S2 genus Pinus"),
    ("s5_pinus", "S5 within genus"),
    ("S3_block0", "S3 source block 0"),
    ("S3_block1", "S3 source block 1"),
    ("S3_block2", "S3 source block 2"),
    ("S3_block3", "S3 source block 3"),
    ("S3_block4", "S3 source block 4"),
    ("S4_fagaceae", "S4 family Fagaceae"),
    ("S4_betulaceae", "S4 family Betulaceae"),
]


def summarise(root: Path, name: str) -> dict | None:
    d = root / name
    if not d.exists():
        return None
    frames = []
    for f in sorted(d.glob("f*/osr_summary.csv")):
        x = pd.read_csv(f)
        x = x.rename(columns={x.columns[0]: "scorer"})
        frames.append(x)
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    g = a.groupby("scorer").mean(numeric_only=True)
    return {
        "acc": float(g["known_acc"].mean()),
        "auroc": float(g["auroc"].mean()),
        "auroc_best": float(g["auroc"].max()),
        "accept": float(g["unknown_accept_rate"].mean()),
        "genus": float(g["genus_recall_unknown"].mean()),
        "family": float(g["family_recall_unknown"].mean()),
        "folds": len(frames),
    }


def main() -> int:
    rows = []
    for key, label in PARTITIONS:
        v = summarise(VIEWS, key)
        p = summarise(PC, key)
        if v is None and p is None:
            continue
        rows.append({
            "partition": label,
            "views_acc": v["acc"] if v else np.nan,
            "pc_acc": p["acc"] if p else np.nan,
            "views_auroc": v["auroc"] if v else np.nan,
            "pc_auroc": p["auroc"] if p else np.nan,
            "views_accept": v["accept"] if v else np.nan,
            "pc_accept": p["accept"] if p else np.nan,
            "views_family": v["family"] if v else np.nan,
            "pc_family": p["family"] if p else np.nan,
            "views_folds": v["folds"] if v else 0,
            "pc_folds": p["folds"] if p else 0,
        })
    d = pd.DataFrame(rows).sort_values("views_acc", ascending=False)

    lines = ["=" * 96,
             "BACKBONE x PARTITION MATRIX",
             "views = multi-view ConvNeXt-Tiny (768-d, 192x192 renders)",
             "pc    = PointNet-style on raw 1,024-point clouds (2,048-d)",
             "=" * 96, ""]
    hdr = (f"{'partition':<22}{'views acc':>10}{'pc acc':>9}{'diff':>8}"
           f"{'views AUROC':>13}{'pc AUROC':>10}{'views accept':>14}{'pc accept':>11}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for _, r in d.iterrows():
        diff = r.pc_acc - r.views_acc if pd.notna(r.pc_acc) else np.nan
        lines.append(
            f"{r.partition:<22}{r.views_acc:>10.3f}{r.pc_acc:>9.3f}{diff:>+8.3f}"
            f"{r.views_auroc:>13.3f}{r.pc_auroc:>10.3f}"
            f"{r.views_accept:>13.1%}{r.pc_accept:>11.1%}")
    lines.append("")

    # rank correlation between the two backbones on accuracy
    ok = d.dropna(subset=["views_acc", "pc_acc"])
    if len(ok) >= 4:
        # Pearson on ranks is Spearman; scipy is not installed in this venv
        rho = ok["views_acc"].rank().corr(ok["pc_acc"].rank())
        pear = ok["views_acc"].corr(ok["pc_acc"])
        lines.append(f"partitions comparable        : {len(ok)}")
        lines.append(f"Spearman rank correlation    : {rho:.3f}")
        lines.append(f"Pearson correlation          : {pear:.3f}")
        lines.append(f"mean accuracy gap (pc - views): {(ok.pc_acc - ok.views_acc).mean():+.3f}")
    lines.append("")
    lines.append("folds: views " + ", ".join(f"{r.partition}={r.views_folds}" for _, r in d.iterrows()
                                               if r.views_folds) )
    lines.append("       pc    " + ", ".join(f"{r.partition}={r.pc_folds}" for _, r in d.iterrows()
                                               if r.pc_folds))

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
