#!/usr/bin/env python3
"""Detect whether external trees are already inside the training set, by render fingerprint.

Name-based checking is what failed on LAUTx: the dataset is simply one of the 25 sources
inside FOR-species20K, and the model scored 0.977 on it. But a source can also be a
*partial* subset, and then the useful move is not to discard the whole collection but to
find which trees were used for training and drop exactly those.

Since every tree is rendered by the same code into the same normalised frame, the same
point cloud produces a near-identical image. So each external tree is compared against
all 17,704 training renders and the nearest-neighbour distance decides membership. True
duplicates sit at essentially zero distance; unrelated trees do not, and the gap is
usually unmistakable.

    python code/find_duplicates.py --views data/_ext/lautx_views --meta data/_ext/lautx_meta.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FOR_VIEWS = Path("data/views_fix/views.npy")   # FIXED renders (validity mask corrected)
FOR_META = Path("data/views/meta.csv")
OUT = Path("results/tables/duplicate_audit_fix.txt")


def descriptor(arr, rows, size=12):
    """Flatten a (N,4,3,192,192) uint8 stack into a coarse fingerprint per tree."""
    out = np.zeros((len(rows), 4 * 3 * size * size), dtype=np.float32)
    step = 192 // size
    for i, r in enumerate(rows):
        v = arr[r][:, :, ::step, ::step][:, :, :size, :size].astype(np.float32)
        out[i] = v.reshape(-1)
    # scale so that distances are comparable across trees
    out /= 255.0
    return out


DEGENERATE_INK = 0.002   # renders below 0.2% non-zero pixels carry no distinctive signal
MAX_PIXEL_DIFF = 8       # full-resolution tolerance for calling two renders identical


def ink_of(arr, rows) -> np.ndarray:
    """Fraction of non-zero pixels per render; near-zero means an almost empty image."""
    out = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i] = float((np.asarray(arr[r]) > 0).mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--label", type=str, default="external")
    ap.add_argument("--save", type=Path, default=None,
                    help="write per-tree match distance to this CSV")
    args = ap.parse_args()

    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ext = np.load(args.views / "views.npy", mmap_mode="r")
    ext_meta = pd.read_csv(args.meta)
    for_views = np.load(FOR_VIEWS, mmap_mode="r")
    for_meta = pd.read_csv(FOR_META)

    n_ext = min(len(ext), len(ext_meta))
    ext_rows = list(range(n_ext))
    D_ext = descriptor(ext, ext_rows)
    print(f"[ext] {n_ext} trees, fingerprint {D_ext.shape}")

    # training side, built in chunks to keep memory bounded
    keep = for_meta.ok.astype(bool).to_numpy() if "ok" in for_meta else np.ones(len(for_meta), bool)
    tr_rows = np.where(keep)[0]
    print(f"[for] {len(tr_rows)} training renders")

    best_d = np.full(n_ext, np.inf, dtype=np.float32)
    best_i = np.full(n_ext, -1, dtype=np.int64)
    CH = 2000
    for s in range(0, len(tr_rows), CH):
        chunk = tr_rows[s:s + CH]
        D_chunk = descriptor(for_views, chunk)
        # squared L2 between external and this chunk
        d2 = ((D_ext ** 2).sum(1)[:, None]
              + (D_chunk ** 2).sum(1)[None, :]
              - 2 * D_ext @ D_chunk.T)
        ink_chunk = ink_of(for_views, chunk)
        d2[:, ink_chunk < DEGENERATE_INK] = np.inf
        j = d2.argmin(1)
        v = d2[np.arange(n_ext), j]
        upd = v < best_d
        best_d[upd] = v[upd]
        best_i[upd] = chunk[j[upd]]
        if (s // CH) % 2 == 0:
            print(f"   scanned {min(s + CH, len(tr_rows))}/{len(tr_rows)}", flush=True)

    dist = np.sqrt(np.maximum(best_d, 0))
    order = np.sort(dist)
    print()
    print("distance quantiles: " + "  ".join(
        f"p{q}={np.quantile(dist, q/100):.5f}" for q in (1, 5, 10, 25, 50, 75, 90)))

    # a duplicate should be essentially identical; pick the cut from the distribution
    cut = 1e-3
    nominal = dist < cut

    # confirm each nominal match against the full-resolution images: a fingerprint that
    # downsamples to the same coarse grid can still come from two different clouds
    dup = np.zeros(n_ext, dtype=bool)
    collisions = []
    for i in np.where(nominal)[0]:
        a = np.asarray(ext[i], dtype=np.int16)
        b = np.asarray(for_views[best_i[i]], dtype=np.int16)
        diff = np.abs(a - b)
        if diff.max() <= MAX_PIXEL_DIFF:
            dup[i] = True
        else:
            collisions.append((i, float(diff.mean()), int(diff.max())))

    print(f"nominal matches at distance < {cut}: {int(nominal.sum())} / {n_ext}")
    print(f"  confirmed as duplicates           : {int(dup.sum())}")
    print(f"  resolved as fingerprint collisions: {len(collisions)}")

    lines = []
    w = lines.append
    w("=" * 78)
    w(f"DUPLICATE AUDIT: {args.label}")
    w("=" * 78)
    w(f"external trees          : {n_ext}")
    w(f"training renders        : {len(tr_rows)}")
    w("")
    w("nearest-training-render distance quantiles:")
    for q in (1, 5, 10, 25, 50, 75, 90, 99):
        w(f"   p{q:<3} {np.quantile(dist, q/100):.5f}")
    w("")
    w(f"nominal matches at distance < {cut}  : {int(nominal.sum())}")
    w(f"  confirmed at full resolution      : {int(dup.sum())}")
    w(f"  resolved as collisions            : {len(collisions)}")
    w(f"looks independent                   : {int((~dup).sum())}")
    if collisions:
        w("")
        w("collisions (same coarse fingerprint, different images):")
        for i, mad, mx in collisions:
            w(f"   {ext_meta.iloc[i]['filename']:<38} mean |diff| {mad:.2f}  max {mx}")
    if args.save:
        pd.DataFrame({
            "treeID": ext_meta.treeID.to_numpy()[:n_ext],
            "nearest_for_treeID": for_meta.treeID.to_numpy()[best_i],
            "distance": dist,
            "nominal": nominal,
            "duplicate": dup,
        }).to_csv(args.save, index=False, encoding="utf-8")
        w(f"per-tree result written to {args.save}")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
