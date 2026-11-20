#!/usr/bin/env python3
"""Build S5 splits: unknown species INSIDE a known genus.

Why this split exists
---------------------
Genus-level fallback ("I don't know the species, but it is a Quercus") only makes
sense when the unknown species belongs to a genus that IS in the training set.
Holding out a whole genus (S2) therefore cannot test fallback at all -- the genus
itself is unseen.  S5 keeps K species of a multi-species genus as known and makes
the remaining species unknown, while every other genus stays known.

The site trap
-------------
If an unknown species comes from datasets the training set already contains, the
model has effectively seen that site, and "same genus" becomes a same-site
artifact.  In FOR-species20K this is the default situation (e.g. Quercus_ilex and
Quercus_faginea both come 100% from Owen_2021).  Two modes:

    default            keep every species of the genus (the overlap is reported anyway)
    --disjoint-sites   DROP the unknown species that share any dataset with the
                       known species of the same genus, and exclude them entirely
                       (they are NOT moved into the training set either)

Usage
-----
    python code/make_splits_s5.py \
        --metadata data/metadata/tree_metadata_dev.csv \
        --genus Pinus Quercus --keep-known 2 --folds 3 --out splits
    # the clean variant
    python code/make_splits_s5.py --genus Pinus Quercus --disjoint-sites --out splits_clean
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def assign_groups(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Greedy balanced assignment of whole datasets to folds."""
    sizes = df.groupby("dataset").size().sort_values(ascending=False)
    rng = np.random.default_rng(seed)
    tie = {d: rng.random() for d in sizes.index}
    order = sorted(sizes.index, key=lambda d: (-sizes[d], tie[d]))
    load = {f: 0 for f in range(n_folds)}
    fold_of = {}
    for d in order:
        f = min(load, key=lambda k: (load[k], k))
        fold_of[d] = f
        load[f] += int(sizes[d])
    return df["dataset"].map(fold_of)


def site_overlap(meta: pd.DataFrame, sp: str, known_site: set) -> tuple:
    d = meta[meta.species == sp]
    if not len(d):
        return float("nan"), []
    ds = set(d.dataset)
    ov = sorted(ds & known_site)
    share = float(d.dataset.isin(ov).mean())
    return share, ov


def build(meta: pd.DataFrame, genus: str, keep_known: int, n_folds: int, seed: int,
          disjoint_sites: bool, fold_mode: str = "random"):
    sub = meta[meta.genus == genus]
    per_species = sub.groupby("species").size().sort_values(ascending=False)
    if len(per_species) <= keep_known:
        return None

    known_sp = list(per_species.index[:keep_known])
    unknown_sp = list(per_species.index[keep_known:])
    known_site = set(sub[sub.species.isin(known_sp)].dataset)

    dropped = []
    if disjoint_sites:
        kept = []
        for sp in unknown_sp:
            share, ov = site_overlap(meta, sp, known_site)
            if ov:
                dropped.append({"species": sp,
                                "n": int(len(meta[meta.species == sp])),
                                "shared_datasets": ov,
                                "frac": round(share, 3)})
            else:
                kept.append(sp)
        unknown_sp = kept

    excluded = {d["species"] for d in dropped}
    known = meta[~meta.species.isin(unknown_sp) & ~meta.species.isin(excluded)].copy()
    unknown = meta[meta.species.isin(unknown_sp)].copy()
    # S5 asks about unknown SPECIES, so the validation split must be tree-level random.
    # Grouping by dataset here made every validation fold cross-site and dropped
    # accuracy from 0.45 to 0.14, which confounds the species question with a site one.
    if fold_mode == "grouped":
        known["fold"] = assign_groups(known, n_folds, seed)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(known))
        col = np.empty(len(known), dtype=int)
        col[perm] = np.arange(len(known)) % n_folds
        known["fold"] = col

    folds = []
    for f in range(n_folds):
        tr = known[known.fold != f]
        va = known[known.fold == f]
        folds.append({
            "fold": f,
            "train": sorted(tr.treeID.astype(int).tolist()),
            "val": sorted(va.treeID.astype(int).tolist()),
            "val_datasets": sorted(va.dataset.unique().tolist()),
        })

    return dict(genus=genus, known_sp=known_sp, unknown_sp=unknown_sp, dropped=dropped,
                known=known, unknown=unknown, folds=folds, per_species=per_species,
                known_site=known_site)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build S5 (within-genus holdout) splits")
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/tree_metadata_dev.csv"))
    ap.add_argument("--genus", nargs="+", default=["Pinus", "Quercus"])
    ap.add_argument("--keep-known", type=int, default=2)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--fold-mode", choices=["random", "grouped"], default="random",
                    help="random = tree-level split (species question); grouped = by dataset")
    ap.add_argument("--disjoint-sites", action="store_true",
                    help="drop unknown species whose datasets are already seen by the "
                         "known species of the same genus (they are removed entirely)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    meta = pd.read_csv(args.metadata)
    meta["genus"] = meta["genus"].fillna(meta["species"].str.split("_").str[0])

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 74)
    emit("S5 SPLITS  --  unknown species inside a known genus")
    emit("=" * 74)
    emit(f"metadata      : {args.metadata}  ({len(meta):,} trees, {meta.dataset.nunique()} datasets)")
    emit(f"rule          : keep the {args.keep_known} largest species of the genus as KNOWN;")
    emit("                all other genera stay known; remaining species of the genus = UNKNOWN")
    emit(f"disjoint-sites: "
         f"{'ON  (drop unknown species sharing a dataset with same-genus known)' if args.disjoint_sites else 'off'}")
    emit("")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for g in args.genus:
        b = build(meta, g, args.keep_known, args.folds, args.seed, args.disjoint_sites, args.fold_mode)
        if b is None:
            emit(f"[skip] {g}: fewer than {args.keep_known + 1} species")
            continue

        emit("-" * 74)
        emit(f"GENUS {g}")
        emit("-" * 74)
        emit(f"  known species   : {', '.join(b['known_sp'])}")
        emit(f"  unknown species : {', '.join(b['unknown_sp']) if b['unknown_sp'] else '(none left)'}")
        if b["dropped"]:
            emit("  DROPPED (site shared with same-genus known species):")
            for d in b["dropped"]:
                emit(f"    {d['species']:<22} n={d['n']:>5}  {d['frac']:.0%} of trees at "
                     f"{', '.join(d['shared_datasets'])}")
        emit("")

        if not b["unknown_sp"]:
            emit("  -> no site-disjoint unknown species remains; S5 not feasible for this genus.")
            emit("")
            manifest[g] = {"feasible": False, "dropped": [d["species"] for d in b["dropped"]]}
            continue

        emit(f"  known trees     : {len(b['known']):,}")
        emit(f"  unknown trees   : {len(b['unknown']):,}")
        emit("")
        emit("  per-unknown-species:")
        for sp in b["unknown_sp"]:
            share, ov = site_overlap(meta, sp, b["known_site"])
            d = meta[meta.species == sp]
            flag = "  <-- SITE OVERLAP" if ov else "  (site-disjoint)"
            emit(f"    {sp:<22} n={len(d):>4}  {d.dataset.nunique()} dataset(s)  "
                 f"overlap={share:.0%}{flag}")
        emit("")
        emit("  fold composition (known trees only):")
        for fd in b["folds"]:
            emit(f"    fold {fd['fold']}: train {len(fd['train']):>6,}  val {len(fd['val']):>6,}")
        emit("")

        payload = {
            "meta": {
                "split": "S5",
                "genus": g,
                "known_species": b["known_sp"],
                "unknown_species": b["unknown_sp"],
                "dropped_species": [d["species"] for d in b["dropped"]],
                "disjoint_sites": bool(args.disjoint_sites),
                "keep_known": args.keep_known,
                "n_folds": args.folds,
                "seed": args.seed,
                "grouping": "dataset (site) grouped, known trees only",
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source": str(args.metadata),
            },
            "unknown_treeIDs": sorted(b["unknown"].treeID.astype(int).tolist()),
            "folds": b["folds"],
        }
        path = args.out / f"s5_{g.lower()}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        manifest[g] = {
            "feasible": True,
            "path": str(path),
            "known_n": int(len(b["known"])),
            "unknown_n": int(len(b["unknown"])),
            "unknown_species": b["unknown_sp"],
            "dropped": [d["species"] for d in b["dropped"]],
        }
        emit(f"  -> wrote {path}")
        emit("")

    (args.out / "s5_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (args.out / "s5_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[s5] wrote {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
