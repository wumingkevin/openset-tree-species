#!/usr/bin/env python3
"""Generate every open-set partition in one schema, so one driver can run them all.

Schema (shared with make_splits_s5.py, which already produced S5):

    {
      "meta": {"split", "variant", "known_species", "unknown_species", "n_folds", ...},
      "unknown_treeIDs": [...],
      "folds": [{"fold", "train": [...], "val": [...], "val_datasets": [...]}]
    }

    S1  random species           6 unknown species, 5 folds x 3 seeds
    S2  whole genus              Pinus / Quercus / Acer, one file each
    S3  leave-one-source-out     sources grouped into 5 balanced blocks
    S4  whole family             Fagaceae / Betulaceae / Sapindaceae
    S5  within-genus             already built by make_splits_s5.py (not duplicated here)

Known-tree folds are always grouped by `dataset` so no acquisition site appears in both
training and validation.  This needs only the metadata CSV, so it can run before the
point clouds exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from make_splits_s5 import assign_groups  # noqa: E402

FAMILY = {
    "Abies": "Pinaceae", "Larix": "Pinaceae", "Picea": "Pinaceae",
    "Pinus": "Pinaceae", "Pseudotsuga": "Pinaceae",
    "Fagus": "Fagaceae", "Quercus": "Fagaceae",
    "Betula": "Betulaceae", "Carpinus": "Betulaceae", "Corylus": "Betulaceae",
    "Acer": "Sapindaceae", "Populus": "Salicaceae", "Fraxinus": "Oleaceae",
    "Crataegus": "Rosaceae", "Prunus": "Rosaceae", "Tilia": "Malvaceae",
    "Ulmus": "Ulmaceae", "Eucalyptus": "Myrtaceae", "Euonymus": "Celastraceae",
}


def make_folds(known: pd.DataFrame, n_folds: int, seed: int, mode: str = "random") -> list[dict]:
    """Build train/val folds for the KNOWN trees.

    mode="random"  : split trees at random. Use for the species-level partitions
                     (S1, S2, S4). Site shift must not be mixed into a question about
                     unknown species, or the two effects cannot be told apart -- and
                     mixing them made every validation fold cross-site, which collapsed
                     accuracy from 0.45 to 0.14 in a controlled check.
    mode="grouped" : whole datasets go to one side. Use only for S3, which is the
                     partition that is actually about a change of site.
    """
    known = known.copy()
    if mode == "grouped":
        known["fold"] = assign_groups(known, n_folds, seed)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(known))
        col = np.empty(len(known), dtype=int)
        col[perm] = np.arange(len(known)) % n_folds
        known["fold"] = col
    out = []
    for f in range(n_folds):
        tr, va = known[known.fold != f], known[known.fold == f]
        out.append({"fold": f,
                    "train": sorted(tr.treeID.astype(int).tolist()),
                    "val": sorted(va.treeID.astype(int).tolist()),
                    "val_datasets": sorted(va.dataset.unique().tolist())})
    return out


def write(path: Path, meta: dict, unknown_ids: list[int], folds: list[dict]) -> dict:
    payload = {"meta": meta, "unknown_treeIDs": sorted(int(i) for i in unknown_ids),
               "folds": folds}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "kb": round(path.stat().st_size / 1024),
            "n_unknown": len(unknown_ids),
            "n_known": sum(len(f["train"]) + len(f["val"]) for f in folds[:1]),
            "unknown_species": meta["unknown_species"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate S1-S4 open-set partitions")
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/tree_metadata_dev.csv"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--s1-unknown", type=int, default=6)
    ap.add_argument("--s1-seeds", type=int, default=3)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    meta = pd.read_csv(args.metadata)
    meta["genus"] = meta["genus"].fillna(meta["species"].str.split("_").str[0])
    meta["family"] = meta["genus"].map(FAMILY)
    if meta["family"].isna().any():
        raise SystemExit(f"unmapped genera: {sorted(meta.loc[meta.family.isna(),'genus'].unique())}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {}

    def emit(s=""):
        print(s)

    emit("=" * 74)
    emit("ALL PARTITIONS")
    emit("=" * 74)
    emit(f"metadata: {args.metadata}  ({len(meta):,} trees, {meta.dataset.nunique()} sources)")
    emit("")

    # ------------------------------------------------------------------ S1
    emit("-" * 74)
    emit(f"S1  random species, {args.s1_unknown} unknown, {args.folds} folds x {args.s1_seeds} seeds")
    emit("-" * 74)
    emit("    unknown species are chosen so that the total unknown tree count lands near a")
    emit("    common target; a plain random draw swings between 1.1k and 4.3k trees and")
    emit("    swamps the seed variance in AUROC.")
    species = sorted(meta.species.unique())
    counts = meta.groupby("species").size()
    target = args.s1_unknown * float(counts.mean())
    for seed in range(args.s1_seeds):
        rng = np.random.default_rng(seed)
        best, best_diff = None, float("inf")
        for _ in range(4000):                      # sample subsets, keep the closest
            cand = rng.choice(species, size=args.s1_unknown, replace=False)
            diff = abs(float(counts[cand].sum()) - target)
            if diff < best_diff:
                best, best_diff = cand, diff
        unknown = sorted(best.tolist())
        known = meta[~meta.species.isin(unknown)]
        unknown_ids = meta[meta.species.isin(unknown)].treeID.tolist()
        folds = make_folds(known, args.folds, seed, mode="random")
        m = {"split": "S1", "variant": f"seed{seed}", "known_species":
             [s for s in species if s not in unknown], "unknown_species": unknown,
             "n_folds": args.folds, "seed": seed,
             "grouping": "dataset (site) grouped; species-stratified holdout",
             "created": stamp}
        info = write(args.out / f"S1_seed{seed}.json", m, unknown_ids, folds)
        manifest[f"S1_seed{seed}"] = info
        emit(f"  seed{seed}: unknown={unknown}")
        emit(f"           {info['n_unknown']:,} unknown trees, "
             f"val {len(folds[0]['val']):,} per fold  -> {info['path']}")
    emit("")

    # ------------------------------------------------------------------ S2
    emit("-" * 74)
    emit("S2  whole genus")
    emit("-" * 74)
    for g in ("Pinus", "Quercus", "Acer"):
        sub = meta[meta.genus == g]
        unknown = sorted(sub.species.unique())
        known = meta[meta.genus != g]
        folds = make_folds(known, 3, 0, mode="random")
        m = {"split": "S2", "variant": g,
             "known_species": sorted(known.species.unique()),
             "unknown_species": unknown, "n_folds": 3, "seed": 0,
             "grouping": "dataset (site) grouped",
             "note": "genus not in label space -> genus fallback is undefined, not zero",
             "created": stamp}
        info = write(args.out / f"S2_{g.lower()}.json", m, sub.treeID.tolist(), folds)
        manifest[f"S2_{g}"] = info
        emit(f"  {g:<9} unknown={len(unknown)} species / {info['n_unknown']:,} trees, "
             f"known {len(known):,} trees -> {info['path']}")
    emit("")

    # ------------------------------------------------------------------ S3
    emit("-" * 74)
    emit(f"S3  leave-one-source-out, sources grouped into {args.folds} balanced blocks")
    emit("-" * 74)
    sizes = meta.groupby("dataset").size().sort_values(ascending=False)
    rng = np.random.default_rng(0)
    tie = {d: rng.random() for d in sizes.index}
    order = sorted(sizes.index, key=lambda d: (-sizes[d], tie[d]))
    load = {f: 0 for f in range(args.folds)}
    block = {}
    for d in order:
        f = min(load, key=lambda k: (load[k], k))
        block[d] = f
        load[f] += int(sizes[d])
    meta["block"] = meta.dataset.map(block)
    for f in range(args.folds):
        held = meta[meta.block == f]
        keep = meta[meta.block != f]
        folds = make_folds(keep, 3, 0, mode="grouped")
        m = {"split": "S3", "variant": f"block{f}",
             "known_species": sorted(keep.species.unique()),
             "unknown_species": sorted(held.species.unique()),
             "n_folds": 3, "seed": 0,
             "grouping": "dataset (site) grouped; held-out sources removed entirely",
             "held_out_sources": sorted(held.dataset.unique().tolist()),
             "note": "unknown set is defined by SOURCE, so species overlap is expected",
             "created": stamp}
        info = write(args.out / f"S3_block{f}.json", m, held.treeID.tolist(), folds)
        manifest[f"S3_block{f}"] = info
        emit(f"  block{f}: holds out {len(m['held_out_sources'])} sources, "
             f"{info['n_unknown']:,} trees -> {info['path']}")
    emit("")

    # ------------------------------------------------------------------ S4
    emit("-" * 74)
    emit("S4  whole family")
    emit("-" * 74)
    emit("    a family that contains only one genus makes the family holdout identical to")
    emit("    that genus holdout, so it is skipped rather than run twice.")
    for fam in ("Fagaceae", "Betulaceae", "Sapindaceae"):
        held = meta[meta.family == fam]
        keep = meta[meta.family != fam]
        n_genera = held.genus.nunique()
        if n_genera < 2:
            emit(f"  [skip] {fam}: only {n_genera} genus "
                 f"({', '.join(sorted(held.genus.unique()))}) -- identical to a genus holdout")
            continue
        folds = make_folds(keep, 3, 0, mode="random")
        m = {"split": "S4", "variant": fam,
             "known_species": sorted(keep.species.unique()),
             "unknown_species": sorted(held.species.unique()),
             "n_folds": 3, "seed": 0,
             "grouping": "dataset (site) grouped",
             "created": stamp}
        info = write(args.out / f"S4_{fam.lower()}.json", m, held.treeID.tolist(), folds)
        manifest[f"S4_{fam}"] = info
        emit(f"  {fam:<12} {len(m['unknown_species'])} species / {info['n_unknown']:,} trees "
             f"-> {info['path']}")
    emit("")

    (args.out / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(f"[done] {len(manifest)} split files + splits_manifest.json in {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
