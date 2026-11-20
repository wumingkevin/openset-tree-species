#!/usr/bin/env python3
"""Dataset and split plumbing for FOR-species20K.

Reads what build_npz.py produced:
    points.npy   (N, K, 3) float32, already centred / ground-aligned / scaled
    meta.csv     one row per array row, with `ok` marking readable trees

Augmentation rules that matter
------------------------------
A tree has a gravity axis, so the only valid rotation is about z.  A full SO(3)
rotation -- the default in almost every ModelNet-style pipeline copied into this
domain -- turns a tree upside down and is physically meaningless here.  Scale jitter
is disabled by default on purpose: absolute size is a real species cue, and shrinking
or growing a tree by 20% fabricates a different species' morphology.

    z-rotation      always allowed
    xy translation  small, harmless
    scale jitter    off by default, small if enabled
    point jitter    small gaussian, plus random resampling of the K points
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"torch is required for the dataset/training layer ({e}).\n"
        f"    pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )


def load_meta(meta_csv: Path) -> pd.DataFrame:
    m = pd.read_csv(meta_csv)
    if "ok" in m.columns:
        m = m[m["ok"].astype(bool)].reset_index(drop=True)
    return m


def classes_from_split(meta: pd.DataFrame, unknown_ids: set[int],
                       train_ids: set[int] | None = None) -> list[str]:
    """Training label space.

    The label space is the species that actually occur in the TRAINING rows, not "every
    species minus the unknown ones". The old rule deleted a species outright as soon as any
    of its trees landed in the unknown set, which on a source-holdout fold removes most of
    the label space: block 1 kept 13 of 33 classes while its training rows contained far
    more, so those rows carried label -1 and cross_entropy aborted with a device-side
    assert. train_views.py was fixed long ago; this trainer was not.
    """
    if train_ids is not None:
        tr = meta[meta.treeID.isin(train_ids)]
        return sorted(tr["species"].unique())
    unk = meta[meta.treeID.isin(unknown_ids)]
    unknown_species = set(unk["species"].unique())
    return sorted(s for s in meta["species"].unique() if s not in unknown_species)


class TreePoints:
    """Holds the memmap and the row/label bookkeeping for a split."""

    def __init__(self, points_path: Path, meta: pd.DataFrame, classes: list[str],
                 root: Path | None = None):
        root = root or points_path.parent
        self.pts = np.load(root / points_path.name, mmap_mode="r")
        self.meta = meta.reset_index(drop=True)
        self.classes = classes
        self.c2i = {c: i for i, c in enumerate(classes)}
        if len(self.meta) != self.pts.shape[0]:
            raise SystemExit(f"meta rows {len(self.meta)} != points {self.pts.shape[0]}")

    def rows_of(self, tree_ids) -> np.ndarray:
        ids = pd.Index(self.meta["treeID"])
        return ids.get_indexer(list(tree_ids))


class TreeDataset(Dataset):
    def __init__(self, store: TreePoints, rows: np.ndarray, labels: np.ndarray,
                 train: bool = False, rot_z: bool = True,
                 jitter: float = 0.005, translate: float = 0.02,
                 scale_jitter: float = 0.0, resample: float = 0.0,
                 seed: int = 0):
        self.store = store
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = train
        self.rot_z = rot_z
        self.jitter = jitter
        self.translate = translate
        self.scale_jitter = scale_jitter
        self.resample = resample
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        x = np.array(self.store.pts[self.rows[i]], dtype=np.float32)   # copy out of memmap
        if self.train:
            if self.rot_z:
                a = self.rng.uniform(0, 2 * np.pi)
                c, s = np.cos(a), np.sin(a)
                x[:, 0], x[:, 1] = c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]
            if self.resample > 0:
                k = len(x)
                keep = self.rng.choice(k, size=int(k * (1 - self.resample)), replace=False)
                x = x[keep]
                extra = self.rng.choice(len(x), size=k - len(x), replace=True)
                x = np.concatenate([x, x[extra]], axis=0)
            if self.translate > 0:
                x[:, :2] += self.rng.uniform(-self.translate, self.translate, (1, 2)).astype(np.float32)
            if self.scale_jitter > 0:
                x = x * (1.0 + self.rng.uniform(-self.scale_jitter, self.scale_jitter))
            if self.jitter > 0:
                x = x + self.rng.normal(0, self.jitter, x.shape).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                int(self.labels[i]) if self.labels is not None else -1)


def build_datasets(points: Path, meta_csv: Path, split_json: Path,
                   fold_index: int = 0, train_aug: bool = True, **aug):
    meta = load_meta(meta_csv)
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    unknown_ids = set(split["unknown_treeIDs"])
    fold = split["folds"][fold_index]
    classes = classes_from_split(meta, unknown_ids, train_ids=set(fold["train"]))
    store = TreePoints(points, meta, classes)

    tr_rows = store.rows_of(fold["train"])
    va_rows = store.rows_of(fold["val"])
    unk_rows = store.rows_of(sorted(unknown_ids))

    def labels(rows):
        sp = store.meta.iloc[rows]["species"].to_numpy()
        return np.array([store.c2i.get(s, -1) for s in sp], dtype=np.int64)

    tr = TreeDataset(store, tr_rows, labels(tr_rows), train=train_aug, **aug)
    va = TreeDataset(store, va_rows, labels(va_rows), train=False)
    unk = TreeDataset(store, unk_rows, np.full(len(unk_rows), -1), train=False)
    return store, tr, va, unk, classes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="sanity check a split against an npz build")
    ap.add_argument("--points", type=Path, default=Path("data/npz/points.npy"))
    ap.add_argument("--meta", type=Path, default=Path("data/npz/meta.csv"))
    ap.add_argument("--split", type=Path, default=Path("splits_clean/s5_pinus.json"))
    a = ap.parse_args()
    store, tr, va, unk, classes = build_datasets(a.points, a.meta, a.split)
    print(f"points      : {store.pts.shape} {store.pts.dtype}")
    print(f"classes     : {len(classes)} -> {classes[:5]} ...")
    print(f"train / val / unknown : {len(tr)} / {len(va)} / {len(unk)}")
    x, y = tr[0]
    print(f"sample      : x{x.shape} y={y}  x-range {x.min():.3f}..{x.max():.3f}")
    print(f"unknown labels all -1 : {(unk.labels == -1).all()}")
