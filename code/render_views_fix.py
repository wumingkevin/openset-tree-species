#!/usr/bin/env python3
"""Render single-tree point clouds into multi-view geometric images (FIXED validity mask).

Why this route
--------------
Four architectures were tried on raw point coordinates and the ceiling was 0.540
validation accuracy (published references: 0.76 for PointNet++ and 0.85 for a
multi-view ConvNeXtV2 on this same benchmark). The multi-view route is the one that is
known to work here, because projecting to an image plane injects a local-texture prior
that a global point pooling throws away.

Full clouds, not the 1024-point sample
--------------------------------------
1024 points spread over a 192x192 grid leaves about 3% of pixels occupied, so the
images would be nearly empty. This reads the .laz files again and renders the whole
cloud (tens of thousands of points per tree).

Three channels per view, after the FOR-species20K multi-view baseline:
    height   max z in the pixel, on a fixed metric scale
    density  log1p of the point count in the pixel
    depth    distance from the pixel to the camera along the view axis

A fixed spatial extent (+-0.60 in normalised units, i.e. 30 m) is used instead of
per-tree fitting, so a tall tree renders large and a short one renders small --
absolute size is a real species cue.

    python code/render_views.py --limit 3 --preview   # sanity check + PNGs
    python code/render_views.py --views 4 --size 192 --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

LAS_SUFFIXES = {".las", ".laz"}
SPAN = 0.30           # fixed half-extent in normalised units (x50 m -> +-15 m)
SCALE = 50.0          # same global divisor as build_npz.py


def index_files(root: Path) -> dict[str, Path]:
    idx = {}
    for dirpath, _d, files in __import__("os").walk(root):
        for fn in files:
            if Path(fn).suffix.lower() in LAS_SUFFIXES:
                idx.setdefault(Path(fn).stem, Path(dirpath) / fn)
    return idx


def camera(azimuth_deg: float, elev_deg: float):
    """Return (right, up, forward) unit vectors for a look-at-the-origin camera."""
    a, e = math.radians(azimuth_deg), math.radians(elev_deg)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    return right, up, fwd


def render_one(xyz: np.ndarray, views, size: int, span: float | None = None) -> np.ndarray:
    """xyz (N,3) normalised -> (V, 3, size, size) uint8.

    span defaults to the tree radius / 0.90, i.e. each tree is fitted to 90% of the
    frame. Image rows grow downwards, so v is flipped when converted to a row index --
    without that the crown renders at the bottom.
    """
    out = np.zeros((len(views), 3, size, size), dtype=np.uint8)
    half = size / 2.0
    if span is None:
        r_xy = np.sqrt((xyz[:, :2] ** 2).sum(1)).max()
        span = max(float(r_xy), float(np.abs(xyz[:, 2]).max()), 1e-6) / 0.90
    for k, (az, el) in enumerate(views):
        right, up, fwd = camera(az, el)
        u = xyz @ right
        v = xyz @ up
        d = xyz @ fwd

        px = np.floor((u / span + 1.0) * half).astype(np.int32)
        py = (size - 1) - np.floor((v / span + 1.0) * half).astype(np.int32)
        ok = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        if not ok.any():
            continue
        px, py, z, dd = px[ok], py[ok], xyz[ok, 2], d[ok]

        dens = np.zeros((size, size), dtype=np.float32)
        np.add.at(dens, (py, px), 1.0)

        hmap = np.full((size, size), -1e3, dtype=np.float32)
        np.maximum.at(hmap, (py, px), z.astype(np.float32))
        hmap[hmap < -1e2] = 0.0

        dmap = np.full((size, size), 1e3, dtype=np.float32)
        np.minimum.at(dmap, (py, px), dd.astype(np.float32))
        dmap[dmap > 1e2] = 0.0

        out[k, 0] = np.clip(((hmap + span) / (span * 2.0)) * 255, 0, 255).astype(np.uint8)
        out[k, 1] = np.clip(np.log1p(dens) / np.log1p(400.0) * 255, 0, 255).astype(np.uint8)
        # FIX: 原来写的是 valid = dmap > 0 —— dmap 是沿视线轴的深度，相机在原点朝外
        # 正投影，约一半点深度为负（且在 20 度仰角下切面近乎水平），于是每张图被削掉
        # 下半截；而 ~valid 会把三个通道一起清零。正确做法是只问"这个像素有没有点"。
        valid = dens > 0
        if valid.any():
            dmin = float(dmap[valid].min())
            out[k, 2] = np.clip((1.0 - (dmap - dmin) / (span * 2.0)) * 255, 0, 255).astype(np.uint8)
        # empty pixels carry no signal in any channel; leaving them non-zero gives the
        # network a bright constant background to key on
        out[k][:, ~valid] = 0
    return out


_G = {}


def _init(root, index, views, size):
    _G.update(index=index, views=views, size=size)


def _work(args):
    import laspy

    row, tree_id, filename = args
    p = _G["index"].get(Path(str(filename)).stem)
    if p is None:
        return row, None, "not found"
    try:
        las = laspy.read(str(p))
        xyz = np.asarray(las.xyz, dtype=np.float64)
    except Exception as e:                                    # noqa: BLE001
        return row, None, f"{type(e).__name__}: {e}"
    if len(xyz) < 32:
        return row, None, f"only {len(xyz)} points"
    xyz = xyz - np.array([xyz[:, 0].mean(), xyz[:, 1].mean(),
                          (xyz[:, 2].min() + xyz[:, 2].max()) / 2.0])
    xyz = xyz / SCALE
    return row, render_one(xyz, _G["views"], _G["size"]), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata/tree_metadata_dev.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/dev"))
    ap.add_argument("--out", type=Path, default=Path("data/views"))
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="write PNG previews")
    args = ap.parse_args()

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    meta = pd.read_csv(args.metadata)
    if args.limit:
        meta = meta.head(args.limit).copy()
    index = index_files(args.root)
    print(f"[paths] {len(index):,} las/laz found, {len(meta):,} trees requested")

    views = [(360.0 * i / args.views, 20.0) for i in range(args.views)]

    if args.preview:
        from PIL import Image
        args.out.mkdir(parents=True, exist_ok=True)
        for _, r in meta.iterrows():
            p = index.get(Path(str(r["filename"])).stem)
            if p is None:
                print(f"  [skip] {r['filename']} not found")
                continue
            import laspy
            xyz = np.asarray(laspy.read(str(p)).xyz, dtype=np.float64)
            xyz = xyz - np.array([xyz[:, 0].mean(), xyz[:, 1].mean(),
                                  (xyz[:, 2].min() + xyz[:, 2].max()) / 2.0])
            xyz = xyz / SCALE
            imgs = render_one(xyz, views, args.size)
            for k in range(len(views)):
                im = Image.fromarray(imgs[k].transpose(1, 2, 0))
                im.save(args.out / f"tree{int(r['treeID']):05d}_v{k}.png")
            nonempty = [(imgs[k, 1] > 0).mean() for k in range(len(views))]
            print(f"  tree {int(r['treeID'])} {r['species']:<24} points={len(xyz):>7,}  "
                  f"occupied px per view: " + ", ".join(f"{x:.1%}" for x in nonempty))
        print(f"[preview] wrote PNGs to {args.out}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    n = len(meta)
    arr = np.lib.format.open_memmap(args.out / "views.npy", mode="w+", dtype=np.uint8,
                                    shape=(n, len(views), 3, args.size, args.size))
    jobs = [(i, int(r["treeID"]), r["filename"]) for i, (_, r) in enumerate(meta.iterrows())]
    failed = []
    t0 = time.time()
    done = 0
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.root), index, views, args.size)) as pool:
        for row, img, err in pool.imap_unordered(_work, jobs, chunksize=8):
            done += 1
            if err:
                failed.append((int(meta.iloc[row]["treeID"]), err))
            else:
                arr[row] = img
            if done % 1000 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done:>6,}/{len(jobs):,}  {el/60:.1f}m  "
                      f"eta {el/done*(len(jobs)-done)/60:.1f}m  failed {len(failed)}", flush=True)
    arr.flush()
    del arr

    m = meta.copy()
    m["row"] = np.arange(n)
    m["ok"] = ~m.treeID.isin({t for t, _ in failed})
    m.to_csv(args.out / "meta.csv", index=False)
    (args.out / "index.json").write_text(json.dumps({
        "views": views, "size": args.size, "span": SPAN, "scale_divisor_m": SCALE,
        "channels": ["height", "density", "depth"], "n_trees": n,
        "n_ok": int(m.ok.sum()), "n_failed": len(failed),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if failed:
        (args.out / "failed.txt").write_text(
            "\n".join(f"{t}\t{e}" for t, e in failed), encoding="utf-8")
    print(f"[done] {int(m.ok.sum()):,}/{n:,} ok, {len(failed)} failed, "
          f"{(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
