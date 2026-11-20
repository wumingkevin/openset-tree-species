#!/usr/bin/env python3
"""Genus-level probability aggregation baseline for the fallback claim.

The paper argues that abstaining to the genus recovers little. The obvious objection, raised
in review, is that the comparison is unfair: the genus answer is obtained by mapping the
predicted species through the taxonomy, which inherits the species head's mistakes, and the
S5 unknown set contains only Pinus, so any classifier that says "Pinus" scores 100% on it.

This builds the fairer baseline the review asks for. Instead of taking the argmax species and
mapping it, it sums the species probabilities over each genus and takes the argmax genus --
a genuine genus-level posterior available from the same model with no extra training. On the
same unknowns it asks: does the aggregated genus do better than the mapped one, and does
either beat the constant prior?

    python code/genus_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/tables_fix/genus_baseline_fix.txt")
PARTITIONS = [("s5_pinus", "S5 within genus"), ("S2_quercus", "S2 Quercus"),
              ("S2_pinus", "S2 Pinus"), ("S1_seed0", "S1 random species")]


def load(name: str):
    """Per-fold arrays live under runs/<split>_f<fold>, not under results/."""
    frames = []
    for d in sorted(Path("runs_fix").glob(f"{name}_f*")):
        if not (d / "logits.npy").exists():
            continue
        meta = pd.read_csv(d / "meta.csv")
        lg = np.load(d / "logits.npy")
        cls = json.loads((d / "classes.json").read_text(encoding="utf-8"))
        frames.append((meta, lg, cls))
    return frames


def genus_of(names):
    out = []
    for s in names:
        g = str(s).split("_")[0]
        out.append(g)
    return np.array(out)


def main() -> int:
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lines = ["=" * 82,
             "GENUS-LEVEL PROBABILITY AGGREGATION vs SPECIES->GENUS MAPPING",
             "=" * 82, ""]
    header = (f"{'partition':<20}{'n unknown':>11}{'mapped':>9}{'aggregated':>12}"
              f"{'const prior':>13}{'species':>10}")
    lines.append(header)
    lines.append("-" * len(header))

    for key, label in PARTITIONS:
        rows = []
        for meta, lg, cls in load(key):
            z = lg - lg.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            genera = genus_of(cls)
            ug = sorted(set(genera))
            gi = {g: i for i, g in enumerate(ug)}

            # rows: unknown trees for this partition
            split = None
            for cand in ("splits", "splits_clean"):
                for name in (key, key.replace("_seed0", "")):
                    f = Path(cand) / f"{name}.json"
                    if f.exists():
                        split = json.loads(f.read_text(encoding="utf-8"))
                        break
                if split is not None:
                    break
            if split is None:
                continue
            unk = set(split["unknown_treeIDs"])
            is_u = meta.treeID.isin(unk).to_numpy()
            if not is_u.any():
                continue

            true_g = genus_of(meta.loc[is_u, "species"].to_numpy())
            pred_sp = np.array(cls)[p[is_u].argmax(1)]
            mapped = genus_of(pred_sp)

            # aggregated posterior: sum species probabilities within each genus
            G = np.zeros((is_u.sum(), len(ug)))
            for j, g in enumerate(genera):
                if g in gi:
                    G[:, gi[g]] += p[is_u][:, j]
            aggregated = np.array(ug)[G.argmax(1)]

            # constant prior: always answer the most frequent known genus
            known_g = genus_of(meta.loc[~is_u, "species"].to_numpy())
            vals, cnts = np.unique(known_g, return_counts=True)
            prior = vals[cnts.argmax()]
            const = np.full(len(true_g), prior)

            rows.append({
                "n": int(is_u.sum()),
                "mapped": float((mapped == true_g).mean()),
                "aggregated": float((aggregated == true_g).mean()),
                "const": float((const == true_g).mean()),
                "species": float((pred_sp == meta.loc[is_u, "species"].to_numpy()).mean()),
            })
        if not rows:
            continue
        d = pd.DataFrame(rows).mean(numeric_only=True)
        lines.append(f"{label:<20}{int(d.n):>11}{d.mapped:>9.3f}{d.aggregated:>12.3f}"
                     f"{d.const:>13.3f}{d.species:>10.3f}")

    lines += ["", "mapped      = argmax species, then map to genus (what the paper reports)",
              "aggregated  = sum species probabilities within genus, argmax (fair genus head)",
              "const prior = always answer the most frequent genus present in training",
              "species     = species accuracy on the same unknowns"]
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

