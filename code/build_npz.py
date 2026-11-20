#!/usr/bin/env python3
"""Convert FOR-species20K .las/.laz trees into a flat float32 array for training.

Why a flat array instead of one .npz per tree
---------------------------------------------
17,707 x 1024 x 3 x 4 B is about 218 MB.  One file per tree means ~17.7 k tiny files
and random reads at every batch, which is far slower than a single memmap.  So the
script writes:

    <out>/points.npy    float32 (N, k, 3)   np.load(mmap_mode='r') for training
    <out>/meta.csv      one row per tree, in exactly the same order as points.npy
    <out>/index.json    the parameters used (k, scale, sampler, seed, source)
    <out>/failed.txt    trees that could not be read, with the reason

Geometry conventions -- do not change these casually
---------------------------------------------------
    xy     : subtract the xy centroid
    z      : subtract the z minimum (canonical pose; for ULS trees the minimum is
             the crown bottom, not the ground -- acceptable, but say so in the paper)
    scale  : divide everything by a FIXED global constant (default 50 m)

Why a global constant and not tree_H: dividing by tree_H would rescale every tree to
unit height, which destroys absolute size -- one of the strongest morphological cues
for species.  The dataset's own documentation also warns that tree_H is only the z
range of the cloud and is not a real tree height.  tree_H is kept as a separate
scalar feature in meta.csv instead.

Usage
-----
    # 0. check that the files can be found before spending an hour
    python code/build_npz.py --metadata data/metadata/tree_metadata_dev.csv \
        --root data/dev --out data/npz --dry-run

    # 1. the real run (8 workers)
    python code/build_npz.py --metadata data/metadata/tree_metadata_dev.csv \
        --root data/dev --out data/npz --workers 8

    # 2. no LAS files around? exercise the whole pipeline on synthetic clouds
    python code/build_npz.py --demo --limit 200 --out data/npz_demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

LAS_SUFFIXES = {".las", ".laz"}


# ------------------------------------------------------------------ path lookup
def index_files(root: Path) -> dict[str, Path]:
    """basename -> path, so the zip's internal layout does not matter."""
    idx: dict[str, Path] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in LAS_SUFFIXES:
                # key by STEM, not by full name: the metadata records "/train/00070.las"
                # while the archive actually ships 00070.laz, so matching on the full
                # name with its extension would fail for every single tree.
                idx.setdefault(Path(fn).stem, Path(dirpath) / fn)
    return idx


def resolve(row, index: dict[str, Path]) -> Path | None:
    fn = str(row["filename"])
    return index.get(Path(fn).stem)


# --------------------------------------------------------------------- sampling
def fps(xyz: np.ndarray, k: int, pre: int = 20000, seed: int = 0) -> np.ndarray:
    """Farthest point sampling.  Pre-thin with a random draw first: plain FPS on a
    million-point cloud would be ~1e9 distance updates per tree."""
    rng = np.random.default_rng(seed)
    n = len(xyz)
    if n > pre:
        xyz = xyz[rng.choice(n, pre, replace=False)]
        n = pre
    if n <= k:
        idx = rng.choice(n, k, replace=True) if n < k else np.arange(k)
        return xyz[idx]
    sel = np.empty(k, dtype=np.int64)
    d = np.full(n, np.inf)
    sel[0] = int(rng.integers(n))
    last = xyz[sel[0]]
    for i in range(1, k):
        d = np.minimum(d, ((xyz - last) ** 2).sum(1))
        sel[i] = int(d.argmax())
        last = xyz[sel[i]]
    return xyz[sel]


def sample_points(xyz: np.ndarray, k: int, sampler: str, seed: int) -> np.ndarray:
    if sampler == "fps":
        return fps(xyz, k, seed=seed)
    n = len(xyz)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, k, replace=n < k)
    return xyz[idx]


def normalize(xyz: np.ndarray, scale: float) -> np.ndarray:
    out = xyz.astype(np.float64, copy=True)
    out[:, 0] -= out[:, 0].mean()
    out[:, 1] -= out[:, 1].mean()
    out[:, 2] -= out[:, 2].min()
    out /= float(scale)
    return out.astype(np.float32)


# ----------------------------------------------------------------------- worker
_G = {}


def _init(root: str, index: dict, k: int, sampler: str, scale: float, seed: int):
    _G.update(index=index, k=k, sampler=sampler, scale=scale, seed=seed)
    global _LASPY_OK
    try:
        import laspy  # noqa: F401
        _LASPY_OK = True
    except Exception:
        _LASPY_OK = False


_LASPY_OK = False


def _work(args):
    import laspy

    row_idx, tree_id, filename = args
    path = _G["index"].get(Path(str(filename)).stem)
    if path is None:
        return row_idx, None, f"file not found: {filename}"
    try:
        las = laspy.read(str(path))
        xyz = np.asarray(las.xyz, dtype=np.float64)
    except Exception as e:  # noqa: BLE001
        return row_idx, None, f"read error: {type(e).__name__}: {e}"
    if xyz.ndim != 2 or xyz.shape[0] < 8:
        return row_idx, None, f"degenerate cloud: shape={xyz.shape}"
    pts = sample_points(xyz, _G["k"], _G["sampler"], _G["seed"] + int(tree_id))
    return row_idx, normalize(pts, _G["scale"]), ""


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="LAS/LAZ -> flat npz array")
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/tree_metadata_dev.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/dev"), help="where the .las/.laz live")
    ap.add_argument("--out", type=Path, default=Path("data/npz"))
    ap.add_argument("--points", type=int, default=1024)
    ap.add_argument("--sampler", choices=["random", "fps"], default="random")
    ap.add_argument("--scale", type=float, default=50.0, help="global divisor (metres)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count()-1")
    ap.add_argument("--limit", type=int, default=0, help="only the first N trees (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="resolve paths only, write nothing")
    ap.add_argument("--demo", action="store_true", help="synthetic clouds, no LAS needed")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    print("=" * 72)
    print("build_npz  --  LAS/LAZ -> flat float32 array")
    print("=" * 72)

    if args.demo:
        n = args.limit or 200
        rng = np.random.default_rng(0)
        meta = pd.DataFrame({
            "treeID": np.arange(n) + 1,
            "species": rng.choice(["Picea_abies", "Pinus_sylvestris", "Fagus_sylvatica"], n),
            "genus": "Demo",
            "dataset": "demo",
            "data_type": rng.choice(["TLS", "ULS"], n),
            "tree_H": rng.uniform(8, 32, n).round(2),
            "filename": [f"/train/{i:05d}.las" for i in range(n)],
        })
        print(f"[demo] synthesising {n} trees; LAS reading is bypassed, everything else runs")
        index = None
    else:
        meta = pd.read_csv(args.metadata)
        for col in ("treeID", "species", "filename"):
            if col not in meta.columns:
                raise SystemExit(f"metadata is missing column '{col}'")
        if args.limit:
            meta = meta.head(args.limit).copy()
        if not args.root.exists():
            raise SystemExit(f"--root does not exist: {args.root}")
        index = index_files(args.root)
        print(f"[paths] {len(index):,} LAS/LAZ files under {args.root}")
        if not index:
            raise SystemExit("no .las/.laz found -- is dev.zip extracted yet?")
        hits = sum(resolve(r, index) is not None for _, r in meta.iterrows())
        print(f"[paths] {hits:,}/{len(meta):,} metadata rows resolved "
              f"({100 * hits / len(meta):.1f}%)")
        for _, r in meta.head(3).iterrows():
            print(f"        {r['filename']:<24} -> {resolve(r, index)}")

    k = args.points
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    n_total = len(meta)
    arr = np.lib.format.open_memmap(out / "points.npy", mode="w+",
                                    dtype=np.float32, shape=(n_total, k, 3))
    print(f"[out]  points.npy  shape=({n_total:,}, {k}, 3)  "
          f"≈ {n_total * k * 3 * 4 / 1e6:.0f} MB")

    if args.dry_run:
        arr.flush()
        print("[dry-run] wrote an empty array only; nothing was read")
        return 0

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    jobs = [(i, int(r["treeID"]), r["filename"]) for i, (_, r) in enumerate(meta.iterrows())]
    failed = []
    done = 0

    if args.demo:
        rng = np.random.default_rng(1)
        for row_idx, _tid, _fn in jobs:
            xyz = rng.uniform([0, 0, 0], [6, 6, 25], size=(4000, 3))
            pts = sample_points(xyz, k, args.sampler, args.seed + row_idx)
            arr[row_idx] = normalize(pts, args.scale)
            done += 1
    else:
        if not _LASPY_OK:
            # _init sets it in the workers; probe once here for a clear message
            try:
                import laspy  # noqa: F401
            except Exception as e:  # noqa: BLE001
                raise SystemExit(
                    f"laspy is not importable ({e}).\n"
                    f"    pip install laspy lazrs   # lazrs is required for .laz"
                )
        with Pool(workers, initializer=_init,
                  initargs=(str(args.root), index, k, args.sampler, args.scale, args.seed)) as pool:
            for row_idx, pts, err in pool.imap_unordered(_work, jobs, chunksize=16):
                done += 1
                if err:
                    failed.append((int(meta.iloc[row_idx]["treeID"]), err))
                else:
                    arr[row_idx] = pts
                if done % 500 == 0 or done == len(jobs):
                    el = time.time() - t0
                    eta = el / done * (len(jobs) - done)
                    print(f"  {done:>6,}/{len(jobs):,}  elapsed {el/60:5.1f}m  "
                          f"eta {eta/60:5.1f}m  failed {len(failed)}", flush=True)

    arr.flush()
    del arr

    keep = np.ones(n_total, dtype=bool)
    # mark failed rows as NaN so training can filter instead of guessing
    if failed:
        failed_ids = {tid for tid, _ in failed}
        keep = ~meta.treeID.isin(failed_ids).to_numpy()
        a = np.load(out / "points.npy", mmap_mode="r+")
        a[~keep] = np.nan
        a.flush()
        (out / "failed.txt").write_text(
            "\n".join(f"{tid}\t{msg}" for tid, msg in failed), encoding="utf-8")

    meta_out = meta.copy()
    meta_out["row"] = np.arange(n_total)
    meta_out["ok"] = keep
    meta_out.to_csv(out / "meta.csv", index=False, encoding="utf-8")

    (out / "index.json").write_text(json.dumps({
        "source_metadata": str(args.metadata),
        "points_per_tree": k,
        "sampler": args.sampler,
        "scale_divisor_m": args.scale,
        "seed": args.seed,
        "n_trees": int(n_total),
        "n_ok": int(keep.sum()),
        "n_failed": len(failed),
        "geometry": "xy=centroid, z=min, divided by a global constant (NOT by tree_H)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"[done] {int(keep.sum()):,}/{n_total:,} trees ok, {len(failed)} failed, "
          f"{time.time() - t0:.0f}s")
    print(f"[done] {out}/points.npy  {out}/meta.csv  {out}/index.json")
    if failed:
        print(f"[done] see {out}/failed.txt  (failed rows are NaN in points.npy, ok=False in meta.csv)")
    print()
    print("train-time loader:")
    print("    pts = np.load('points.npy', mmap_mode='r')")
    print("    x   = pts[row]                      # (1024, 3) float32, already normalised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
