#!/usr/bin/env python3
"""Plot-clustered bootstrap for the external collections, on the species-labelled subset.

This is the reproduction entry point for the intervals in Table 13. It replaces an earlier
version that granted genus-level credit to trees annotated only to genus (Quercus sp.) and
therefore reported mixed-granularity numbers such as Herby 0.636. The metric here matches the
table: species accuracy is computed only on trees that carry a species-level label, while genus
and family accuracy keep every tree of the district as the denominator.

Fixed configuration, so the endpoints can be reproduced exactly:
  input        results/trees/<district>.csv  (per-tree predictions from ext_evaluate)
  labels       data/_ext/tspl/district_<district>.csv, column genus_only
  filter       species accuracy on rows with genus_only == False
  cluster      the plot column of the per-tree file
  replicates   2,000
  seed         0 (numpy default_rng); clusters resampled in sorted order
  estimator    each replicate is the pooled accuracy over its resampled trees, so a large plot
               contributes in proportion to its size; the point estimate is the observed
               accuracy over all included trees, not the mean of the replicates

    python code/cluster_bootstrap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("code").resolve()))
import metrics as M  # single source of truth for the genus->family map

OUT = Path("results/tables/cluster_bootstrap.txt")
DUMP = Path("results/trees")
TS = Path("data/_ext/tspl")
N_BOOT = 2000
SEED = 0

def genus(names):
    return M.genus_of(names)


def family(names):
    return M.family_of(names)


def boot(df: pd.DataFrame, loop: int = N_BOOT) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    keys = sorted(df["plot"].unique())
    groups = {k: df[df["plot"] == k] for k in keys}
    accs = np.empty(loop)
    for b in range(loop):
        pick = rng.choice(keys, size=len(keys), replace=True)
        d = pd.concat([groups[k] for k in pick], ignore_index=True)
        accs[b] = d.correct.mean()
    return (float(df.correct.mean()), float(np.percentile(accs, 2.5)),
            float(np.percentile(accs, 97.5)))


def load_district(name: str):
    f = DUMP / f"{name}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    meta = TS / f"district_{name}.csv"
    if meta.exists():
        go = pd.read_csv(meta)[["treeID", "genus_only"]]
        d = d.merge(go, on="treeID", how="left")
    else:
        d["genus_only"] = False
    d["genus_only"] = d["genus_only"].fillna(False).astype(bool)
    d["g_ok"] = genus(d.true.values) == genus(d.pred.values)
    d["f_ok"] = family(d.true.values) == family(d.pred.values)
    return d


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lines = ["=" * 98,
             "PLOT-CLUSTERED BOOTSTRAP FOR TABLE 13",
             f"{N_BOOT} replicates, seed {SEED}, clusters resampled in sorted order",
             "species accuracy on species-labelled trees only; genus/family on all trees",
             "=" * 98, ""]
    hdr = (f"{'collection':<18}{'trees':>7}{'sp.-labelled':>14}{'species':>9}{'95% CI':>20}"
           f"{'genus':>8}{'family':>8}{'plots':>7}")
    lines += [hdr, "-" * len(hdr)]

    for name in ("Gorlice", "Herby", "Katrynka", "Milicz", "Piensk", "Belgium"):
        d = load_district(name)
        if d is None:
            continue
        sub = d[~d.genus_only]
        m, lo, hi = boot(sub)
        label = "Belgium Demol" if name == "Belgium" else "Poland " + name
        lines.append(f"{label:<18}{len(d):>7}{len(sub):>14}{m:>9.3f}"
                     f"{f'[{lo:.3f}, {hi:.3f}]':>20}{d.g_ok.mean():>8.3f}"
                     f"{d.f_ok.mean():>8.3f}{d['plot'].nunique():>7}")

    lines += ["", "The species column is the one quoted in Table 13; genus and family use every",
              "tree of the district as the denominator, which is why their counts differ.",
              "Suprasl has no plot-level dump and no interval is reported for it."]
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
