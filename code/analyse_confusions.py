#!/usr/bin/env python3
"""Confusion and error-structure analysis from the all-species reference model.

Sections 5.1 and 5.2 of the manuscript quote per-genus fallback rates and the dominant
same-genus confusion pairs. Those numbers originally came from the released model's
out-of-fold features on the official split, which is a different model from the one the
rest of the paper reports. This recomputes them on the multi-view backbone trained on
all 33 species with a tree-level random split, so the whole paper speaks about one
model.

    python code/analyse_confusions.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path("runs/REF_all33")
META = Path("data/metadata/tree_metadata_dev.csv")
OUT = Path("results/tables/confusion_analysis.txt")

FAMILY = {
    "Abies": "Pinaceae", "Larix": "Pinaceae", "Picea": "Pinaceae",
    "Pinus": "Pinaceae", "Pseudotsuga": "Pinaceae",
    "Fagus": "Fagaceae", "Quercus": "Fagaceae",
    "Betula": "Betulaceae", "Carpinus": "Betulaceae", "Corylus": "Betulaceae",
    "Acer": "Sapindaceae", "Populus": "Salicaceae", "Fraxinus": "Oleaceae",
    "Crataegus": "Rosaceae", "Prunus": "Rosaceae", "Tilia": "Malvaceae",
    "Ulmus": "Ulmaceae", "Eucalyptus": "Myrtaceae", "Euonymus": "Celastraceae",
}


def main() -> int:
    split = json.loads(Path("splits_reference/all_species_f0.json").read_text(encoding="utf-8"))
    val = set(split["folds"][0]["val"])
    meta = pd.read_csv(RUN / "meta.csv")
    cls = json.loads((RUN / "classes.json").read_text(encoding="utf-8"))
    lg = np.load(RUN / "logits.npy")
    src = pd.read_csv(META)[["treeID", "dataset"]].rename(columns={"dataset": "source"})

    c2i = {c: i for i, c in enumerate(cls)}
    y = np.array([c2i.get(s, -1) for s in meta.species])
    is_val = meta.treeID.isin(val).to_numpy() & (y >= 0)
    yk, pred = y[is_val], lg.argmax(1)[is_val]
    names = np.array(cls)
    m = meta[is_val].merge(src, on="treeID", how="left")

    true_sp, pred_sp = names[yk], names[pred]
    true_g = np.array([s.split("_")[0] for s in true_sp])
    pred_g = np.array([s.split("_")[0] for s in pred_sp])
    true_f = np.array([FAMILY.get(g, "?") for g in true_g])
    pred_f = np.array([FAMILY.get(g, "?") for g in pred_g])

    lines = []
    w = lines.append
    w("=" * 78)
    w("REFERENCE MODEL: 33 species, tree-level random 80/20 split")
    w("=" * 78)
    w(f"validation trees evaluated : {len(yk):,}")
    w(f"classes                    : {len(cls)}")
    w("")
    sp_acc = (true_sp == pred_sp).mean()
    g_acc = (true_g == pred_g).mean()
    f_acc = (true_f == pred_f).mean()
    wrong = true_sp != pred_sp
    w(f"species accuracy           : {sp_acc:.3f}")
    w(f"genus accuracy             : {g_acc:.3f}")
    w(f"family accuracy            : {f_acc:.3f}")
    w(f"M2 = P(genus correct | species wrong) : {(true_g[wrong] == pred_g[wrong]).mean():.3f}")
    w(f"M3 = P(family correct | species wrong): {(true_f[wrong] == pred_f[wrong]).mean():.3f}")
    w("")

    # ---- per-genus fallback
    w("-" * 78)
    w("PER-GENUS RECOVERY  (on trees whose species was wrong)")
    w("-" * 78)
    w(f"{'genus':<14}{'n wrong':>9}{'genus right':>13}{'family right':>14}")
    rows = []
    for g in sorted(set(true_g)):
        sel = wrong & (true_g == g)
        if sel.sum() == 0:
            continue
        gr = (true_g[sel] == pred_g[sel]).mean()
        fr = (true_f[sel] == pred_f[sel]).mean()
        rows.append((g, int(sel.sum()), gr, fr))
    for g, n, gr, fr in sorted(rows, key=lambda x: -x[2]):
        w(f"{g:<14}{n:>9}{gr:>13.3f}{fr:>14.3f}")
    w("")

    # ---- dominant same-genus confusion pairs
    w("-" * 78)
    w("DOMINANT SAME-GENUS CONFUSIONS  (true -> predicted, within one genus)")
    w("-" * 78)
    ctab = pd.crosstab(pd.Series(true_sp), pd.Series(pred_sp))
    pairs = []
    for a in ctab.index:
        ga = a.split("_")[0]
        for b in ctab.columns:
            if a == b:
                continue
            n = int(ctab.loc[a, b])
            if n == 0 or b.split("_")[0] != ga:
                continue
            pairs.append((a, b, n))
    pairs.sort(key=lambda x: -x[2])
    w(f"{'true species':<24}{'predicted':<24}{'n':>6}  sources")
    for a, b, n in pairs[:8]:
        sa = set(m.loc[m.species == a, "source"].dropna())
        sb = set(m.loc[m.species == b, "source"].dropna())
        shared = sa & sb
        tag = f"shared: {', '.join(sorted(shared))}" if shared else "disjoint"
        w(f"{a:<24}{b:<24}{n:>6}  {tag}")
    w("")

    # ---- source audit
    w("-" * 78)
    w("SOURCE AUDIT")
    w("-" * 78)
    full = pd.read_csv(META)
    per_sp = full.groupby("species").source.nunique() if "source" in full else None
    g2 = full.groupby("species").dataset.nunique()
    w(f"species in the dataset           : {len(g2)}")
    w(f"species appearing in one source  : {int((g2 == 1).sum())}")
    w(f"genera                           : {full.genus.nunique()}")
    w(f"genera with a single species     : {int((full.groupby('genus').species.nunique() == 1).sum())}")
    w(f"acquisition sources              : {full.dataset.nunique()}")
    w("")
    w("single-source species:")
    for sp, c in g2[g2 == 1].items():
        w(f"   {sp:<26}{full.loc[full.species == sp, 'dataset'].iloc[0]}")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
