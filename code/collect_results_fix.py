#!/usr/bin/env python3
"""Collect every evaluated fold into the tables the manuscript needs.

Reads results/all/<split>/f<fold>/osr_summary.csv for all five partitions, folds the
scorers together, and writes:

    results/tables/table_splits.csv     split x detector, mean and s.d.
    results/tables/table_taxonomy.csv   the neighbourhood table (family/genus recall)
    results/tables/summary.txt          human-readable rendering of both

Structural zeros are marked rather than quietly reported as measurements: on S2_acer
the whole genus and its family are absent from the training labels, so genus- and
family-level recall are undefined there, not zero.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["auroc", "oscr_auc", "fpr_at_95tpr", "osa", "known_acc", "macro_recall", "ece",
           "unknown_accept_rate",
           "genus_recall_unknown", "family_recall_unknown"]
SCORERS = ["msp", "energy", "mahalanobis", "knn"]

# How much of the taxonomic neighbourhood each partition leaves visible, and whether
# the coarse-level metric is even defined.
PARTITION_META = {
    "S1_seed0":   dict(label="S1  random species",      neighbourhood="mixed",
                       family_defined=True),
    "S2_acer":    dict(label="S2  whole genus (Acer)",  neighbourhood="family absent",
                       family_defined=False),
    "S2_quercus": dict(label="S2  whole genus (Quercus)", neighbourhood="1 congener family",
                       family_defined=True),
    "S2_pinus":   dict(label="S2  whole genus (Pinus)", neighbourhood="4 congener genera",
                       family_defined=True),
    "s5_pinus":   dict(label="S5  within genus (Pinus)", neighbourhood="same genus known",
                       family_defined=True),
    "S3_block0":  dict(label="S3  source block 0", neighbourhood="species fixed",
                       family_defined=True),
    "S3_block1":  dict(label="S3  source block 1", neighbourhood="species fixed",
                       family_defined=True),
    "S3_block2":  dict(label="S3  source block 2", neighbourhood="species fixed",
                       family_defined=True),
    "S3_block3":  dict(label="S3  source block 3", neighbourhood="species fixed",
                       family_defined=True),
    "S3_block4":  dict(label="S3  source block 4", neighbourhood="species fixed",
                       family_defined=True),
    "S4_fagaceae": dict(label="S4  whole family (Fagaceae)", neighbourhood="family absent",
                        family_defined=False),
    "S4_betulaceae": dict(label="S4  whole family (Betulaceae)", neighbourhood="family absent",
                          family_defined=False),
}


def load_val_acc(root: Path) -> dict:
    """Per-split held-out validation accuracy, from each fold metrics.json.

    known_acc in osr_summary is computed over train+val known rows, so on the
    source-holdout folds it is dominated by memorised training trees and reads far
    higher than the cross-site number that actually matters. Only the held-out
    validation accuracy answers the site-shift question.
    """
    acc: dict[str, list[float]] = {}
    for p in sorted(glob.glob(str(root / "*" / "metrics.json"))):
        split = Path(p).parent.name.rsplit("_f", 1)[0]
        try:
            v = json.loads(Path(p).read_text(encoding="utf-8")).get("val_acc")
        except Exception:
            continue
        if v is not None:
            acc.setdefault(split, []).append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def load_all(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(str(root / "*" / "f*" / "osr_summary.csv"))):
        split = Path(p).parent.parent.name
        fold = int(Path(p).parent.name.lstrip("f"))
        d = pd.read_csv(p)
        d = d.rename(columns={d.columns[0]: "scorer"})
        d["split"] = split
        d["fold"] = fold
        rows.append(d)
    if not rows:
        raise SystemExit("no osr_summary.csv found")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/all_fix"))
    ap.add_argument("--out", type=Path, default=Path("results/tables_fix"))
    ap.add_argument("--runs", type=Path, default=Path("runs_fix"))
    a, _ = ap.parse_known_args()
    root = a.root
    out = a.out
    runs_dir = a.runs
    out.mkdir(parents=True, exist_ok=True)

    a = load_all(root)
    counts = a.groupby("split").fold.nunique().to_dict()

    # split x detector, mean and sd across folds
    g = a.groupby(["split", "scorer"])[METRICS].agg(["mean", "std"])
    flat = pd.DataFrame({f"{m}_{k}": g[(m, k)] for m in METRICS for k in ("mean", "std")})
    flat["n_folds"] = [counts.get(s, 0) for s in flat.index.get_level_values(0)]
    flat.to_csv(out / "table_splits.csv", encoding="utf-8-sig")

    # taxonomy table: one row per partition, detectors averaged
    tx = a.groupby("split")[["known_acc", "auroc", "unknown_accept_rate",
                             "genus_recall_unknown", "family_recall_unknown"]].mean()
    tx["n_folds"] = [counts.get(s, 0) for s in tx.index]
    tx["family_defined"] = [PARTITION_META.get(s, {}).get("family_defined", True)
                            for s in tx.index]
    tx["neighbourhood"] = [PARTITION_META.get(s, {}).get("neighbourhood", "?")
                           for s in tx.index]
    tx["label"] = [PARTITION_META.get(s, {}).get("label", s) for s in tx.index]
    vstats = load_val_acc(runs_dir)
    tx["val_acc"] = [vstats.get(s, (float("nan"), float("nan")))[0] for s in tx.index]
    tx["val_acc_sd"] = [vstats.get(s, (float("nan"), float("nan")))[1] for s in tx.index]
    tx.to_csv(out / "table_taxonomy.csv", encoding="utf-8-sig")

    lines = []
    lines.append("=" * 104)
    lines.append("PER-PARTITION SUMMARY  (detectors averaged, mean over folds)")
    lines.append("=" * 104)
    lines.append(f"{'partition':<34}{'folds':>6}{'val_acc':>9}{'known_acc':>11}{'auroc':>8}"
                 f"{'accept':>9}{'genus':>9}{'family':>9}  neighbourhood")
    order = ["S1_seed0", "S2_acer", "S2_quercus", "S2_pinus", "s5_pinus",
             "S3_block0", "S3_block1", "S3_block2", "S3_block3", "S3_block4",
             "S4_fagaceae", "S4_betulaceae"]
    for s in order:
        if s not in tx.index:
            continue
        r = tx.loc[s]
        fam = f"{r.family_recall_unknown:9.3f}" if r.family_defined else "   struct"
        va = f"{r.val_acc:9.3f}" if r.val_acc == r.val_acc else "      n/a"
        lines.append(f"{r.label:<34}{int(r.n_folds):>6}{va}{r.known_acc:>11.3f}"
                     f"{r.auroc:>8.3f}{r.unknown_accept_rate:>9.3f}"
                     f"{r.genus_recall_unknown:>9.3f}{fam}  {r.neighbourhood}")
    lines.append("")
    lines.append("val_acc = held-out accuracy (the number that matters); known_acc also")
    lines.append("counts training trees, so on source-holdout folds it is inflated.")
    lines.append("'struct' = the coarse level is absent from the training labels, so the")
    lines.append("metric is undefined, not measured as zero.")

    lines.append("")
    lines.append("=" * 100)
    lines.append("PER-DETECTOR DETAIL")
    lines.append("=" * 100)
    for s in order:
        if s not in counts:
            continue
        lines.append("")
        lines.append(f"-- {PARTITION_META.get(s, {}).get('label', s)}  ({counts[s]} folds)")
        sub = a[a.split == s]
        hdr = f"   {'scorer':<13}" + "".join(f"{m[:11]:>13}" for m in METRICS)
        lines.append(hdr)
        for sc in SCORERS:
            d = sub[sub.scorer == sc]
            if d.empty:
                continue
            fam_ok = PARTITION_META.get(s, {}).get("family_defined", True)
            cells = []
            for m in METRICS:
                if m == "family_recall_unknown" and not fam_ok:
                    cells.append(f"{'struct':>13}")
                else:
                    cells.append(f"{d[m].mean():13.3f}")
            lines.append(f"   {sc:<13}" + "".join(cells))

    text = "\n".join(lines)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    print()
    print(f"[wrote] {out/'table_splits.csv'}  {out/'table_taxonomy.csv'}  {out/'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
