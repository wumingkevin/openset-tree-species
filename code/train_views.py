#!/usr/bin/env python3
"""Train a multi-view CNN on the rendered geometric views.

This is the route the FOR-species20K multi-view baseline took to 0.885, and it exists
here to answer one objection: that the open-set failures measured with a small point
network are an artefact of that network. A backbone that reaches the published range
removes the objection.

Interface is deliberately identical to train_closed.py -- same --split/--fold/--out,
same logits.npy / feats.npy / meta.csv / classes.json dump -- so osr.py consumes either
backbone without changes.

    python code/train_views.py --views data/views/views.npy --meta data/views/meta.csv \
        --split splits_clean/s5_pinus.json --fold 0 --epochs 30 --out runs/mv_s5_f0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision as tv
from torchvision.transforms import functional as TF

# Render statistics measured on the full 17,707-tree render (uint8 / 255).
CH_MEAN = torch.tensor([7.4, 5.1, 8.6]).view(1, 3, 1, 1) / 255.0
CH_STD = torch.tensor([38.4, 28.2, 44.0]).view(1, 3, 1, 1) / 255.0


_MEMMAPS: dict[str, np.ndarray] = {}


def _arr(path: str):
    """Open a render memmap once per process.

    The array must NOT live on the Dataset instance: DataLoader pickles the dataset to
    its workers, and pickling an np.memmap serialises its 7.8 GB of data instead of the
    mapping, which killed the workers with MemoryError.
    """
    if path not in _MEMMAPS:
        _MEMMAPS[path] = np.load(path, mmap_mode="r")
    return _MEMMAPS[path]


class MVViews(Dataset):
    """Rows of a (N, V, 3, H, W) uint8 memmap, labelled from meta.csv."""

    def __init__(self, npy, meta, rows, labels, train=False, flip=True):
        self.path = str(npy)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = train
        self.flip = flip

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        arr = _arr(self.path)
        x = torch.from_numpy(np.array(arr[self.rows[i]])).float() / 255.0  # (V,3,H,W)
        x = (x - CH_MEAN) / CH_STD
        if self.train and self.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[3])
        return x, int(self.labels[i])


class MVConvNeXt(nn.Module):
    """ConvNeXtV2-free variant: torchvision ConvNeXt-Tiny, averaged over views."""

    def __init__(self, n_classes: int, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        w = tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = tv.models.convnext_tiny(weights=w)
        self.features = m.features
        self.avgpool = m.avgpool
        self.norm = m.classifier[0]
        self.feat_dim = m.classifier[2].in_features
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feat_dim, n_classes)

    def forward(self, x, extra=None):
        B, V, C, H, W = x.shape
        z = self.features(x.reshape(B * V, C, H, W))
        z = self.avgpool(z)
        z = self.norm(z)
        g = torch.flatten(z, 1)                      # (B*V, feat)
        logits = self.fc(self.drop(g)).view(B, V, -1).mean(1)
        feats = g.view(B, V, -1).mean(1)
        return logits, feats


def build(views_npy, meta_csv, split_json, fold, train_aug):
    meta_all = pd.read_csv(meta_csv)
    _keep = meta_all.ok.astype(bool).to_numpy()
    meta = meta_all[_keep].reset_index(drop=True)
    # views.npy keeps a row for EVERY tree in the unfiltered meta.csv (17,707 rows,
    # the three unreadable trees are all-zero blocks), so a position in the filtered
    # frame is NOT a row index into views.npy once any row has been dropped. Map
    # filtered positions back to their true views.npy row.
    view_row = np.where(_keep)[0]
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    unknown = set(split["unknown_treeIDs"])
    unk_meta = meta[meta.treeID.isin(unknown)]
    fd = split["folds"][fold]
    idx = pd.Index(meta.treeID)
    tr_rows = idx.get_indexer(fd["train"])
    # Classes must be the species that actually occur in TRAINING, not "all species
    # minus the unknown ones". The old rule deleted a species outright as soon as any
    # one of its trees landed in the unknown set, which on source-holdout folds removed
    # most of the label space: block 1 kept 13 of 33 classes and validation accuracy
    # was identically zero for every epoch.
    classes = sorted(set(meta.iloc[tr_rows[tr_rows >= 0]].species.unique()))
    c2i = {c: i for i, c in enumerate(classes)}
    va_rows = idx.get_indexer(fd["val"])
    un_rows = idx.get_indexer(sorted(unknown))
    for name, arr in (("train", tr_rows), ("val", va_rows), ("unknown", un_rows)):
        if (arr < 0).any():
            print(f"[warn] {name}: {int((arr < 0).sum())} treeIDs missing from meta, dropped")

    def lab(rows):
        r = rows[rows >= 0]
        sp = meta.iloc[r].species.to_numpy()
        return np.array([c2i.get(s, -1) for s in sp], dtype=np.int64)

    tr = MVViews(views_npy, meta, view_row[tr_rows[tr_rows >= 0]], lab(tr_rows),
                 train=train_aug)
    va = MVViews(views_npy, meta, view_row[va_rows[va_rows >= 0]], lab(va_rows))
    un = MVViews(views_npy, meta, view_row[un_rows[un_rows >= 0]],
                 np.full(int((un_rows >= 0).sum()), -1))
    return meta, classes, (tr, va, un), (tr_rows, va_rows, un_rows), view_row


def main() -> int:
    ap = argparse.ArgumentParser(description="multi-view CNN trainer")
    ap.add_argument("--views", type=Path, default=Path("data/views/views.npy"))
    ap.add_argument("--meta", type=Path, default=Path("data/views/meta.csv"))
    ap.add_argument("--split", type=Path, default=Path("splits_clean/s5_pinus.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/mv"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16, help="trees per batch (x4 views)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    meta, classes, (tr, va, un), rows, view_row = build(
        args.views, args.meta, args.split, args.fold, True)
    print(f"[data] train {len(tr)}  val {len(va)}  unknown {len(un)}  classes {len(classes)}  {dev}")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                       num_workers=args.workers, pin_memory=(dev == "cuda"))
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, pin_memory=(dev == "cuda"))

    model = MVConvNeXt(len(classes), pretrained=not args.no_pretrained).to(dev)
    print(f"[model] convnext_tiny  params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def run_epoch(loader, train):
        model.train(train)
        tot = cor = 0
        loss_sum = 0.0
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            keep = y >= 0
            if not keep.all():
                x, y = x[keep], y[keep]
            if len(y) == 0:
                continue
            with torch.set_grad_enabled(train):
                z, _ = model(x)
                loss = F.cross_entropy(z, y, label_smoothing=0.1)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            loss_sum += float(loss.detach()) * len(y)
            cor += int((z.argmax(1) == y).sum())
            tot += len(y)
        if train:
            sched.step()
        return loss_sum / max(tot, 1), cor / max(tot, 1)

    best = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        ltr, atr = run_epoch(dl_tr, True)
        lva, ava = run_epoch(dl_va, False)
        flag = ""
        if ava > best:
            best = ava
            torch.save({"model": model.state_dict(), "classes": classes}, args.out / "best.pt")
            flag = " *"
        print(f"  ep {ep:3d}  train {ltr:.3f}/{atr:.3f}  val {lva:.3f}/{ava:.3f}{flag}  "
              f"{time.time()-t0:.0f}s", flush=True)

    # dump in the order train, val, unknown so osr.py can align by treeID
    ck = torch.load(args.out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    order = np.concatenate([rows[0][rows[0] >= 0], rows[1][rows[1] >= 0], rows[2][rows[2] >= 0]])
    sub = MVViews(args.views, meta, view_row[order], np.zeros(len(order), dtype=np.int64))
    # Dumping with worker processes stalled for over two hours on one split while
    # writing nothing; the same pass takes 99 s with workers=0 and a small batch.
    dl = DataLoader(sub, batch_size=8, shuffle=False, num_workers=0)

    logits, feats = [], []
    with torch.no_grad():
        for x, _ in dl:
            z, g = model(x.to(dev))
            logits.append(z.float().cpu().numpy())
            feats.append(g.float().cpu().numpy())
    logits = np.concatenate(logits)
    feats = np.concatenate(feats)

    np.save(args.out / "logits.npy", logits)
    np.save(args.out / "feats.npy", feats)
    (args.out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False),
                                           encoding="utf-8")
    meta.iloc[order].to_csv(args.out / "meta.csv", index=False)
    (args.out / "metrics.json").write_text(json.dumps(
        {"val_acc": best, "arch": "mv_convnext_tiny", "epochs": args.epochs,
         "n_train": len(tr), "n_val": len(va), "n_unknown": len(un),
         "split": str(args.split), "fold": args.fold}, indent=2), encoding="utf-8")

    print(f"[best] val {best:.4f}")
    print(f"[dump] logits {logits.shape}  feats {feats.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
