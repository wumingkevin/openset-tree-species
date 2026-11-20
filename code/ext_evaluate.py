#!/usr/bin/env python3
"""Zero-shot transfer evaluation: FOR-species20K model applied to the Austrian dataset.

No retraining and no adaptation. The 33-class model trained on FOR-species20K is applied
to 515 Austrian personal-laser-scanner trees, and we report species / genus / family
accuracy plus the confusion structure. The point is not to beat a number; it is to see
whether the source-shift collapse measured in S3 reproduces on a dataset from a
different country, a different campaign and a different scanner.

    python code/ext_evaluate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_views import MVConvNeXt, MVViews
from metrics import FAMILY

DEFAULT_OUT = Path("results/tables/ext_validation.txt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("runs/REF_all33"))
    ap.add_argument("--views", type=Path, default=Path("data/_ext/views/views.npy"))
    ap.add_argument("--meta", type=Path, default=Path("data/_ext/views/meta.csv"))
    ap.add_argument("--trees", type=Path, default=Path("data/_ext/lautx_meta.csv"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dump", type=Path, default=None,
                    help="write one row per tree with the prediction")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path; defaults to a name derived from --trees")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cls = json.loads((args.model / "classes.json").read_text(encoding="utf-8"))
    ck = torch.load(args.model / "best.pt", map_location=dev, weights_only=False)
    model = MVConvNeXt(len(cls), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    vmeta = pd.read_csv(args.meta)
    trees = pd.read_csv(args.trees)
    order = vmeta.reset_index(drop=True)
    ds = MVViews(args.views, vmeta, np.arange(len(order)),
                 np.zeros(len(order), dtype=np.int64))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    probs = []
    with torch.no_grad():
        for x, _ in dl:
            z, _ = model(x.to(dev))
            probs.append(torch.softmax(z.float(), 1).cpu().numpy())
    probs = np.concatenate(probs)

    # align by row order: views meta was written from lautz_meta order
    n = min(len(probs), len(trees))
    probs, trees = probs[:n], trees.iloc[:n].reset_index(drop=True)
    names = np.array(cls)
    pred = names[probs.argmax(1)]
    true = trees.species.to_numpy()
    genus_only = trees.genus_only.astype(bool).to_numpy()

    tg = np.array([s.split("_")[0] for s in true])
    pg = np.array([s.split("_")[0] for s in pred])
    tf = np.array([FAMILY.get(g, "?") for g in tg])
    pf = np.array([FAMILY.get(g, "?") for g in pg])

    sp_ok = pred == true
    if args.dump is not None:
        plot_col = trees["plot"].to_numpy() if "plot" in trees else np.array(["?"] * len(true))
        pd.DataFrame({"treeID": order.treeID.to_numpy()[:len(true)], "plot": plot_col[:len(true)],
                      "true": true, "pred": pred, "correct": sp_ok}).to_csv(
            args.dump, index=False, encoding="utf-8")
        print(f"[dump] per-tree predictions -> {args.dump}")
    # for genus-only labels (oak) any congeneric answer counts at species level
    for i in np.where(genus_only)[0]:
        sp_ok[i] = pg[i] == tg[i]

    lines = []
    w = lines.append
    w("=" * 78)
    w(f"ZERO-SHOT TRANSFER: FOR-species20K  ->  {args.trees.stem}")
    w("=" * 78)
    w(f"trees evaluated        : {n}")
    w(f"model                  : multi-view ConvNeXt-Tiny, {len(cls)} classes, no adaptation")
    w(f"species accuracy       : {sp_ok.mean():.3f}")
    w(f"genus accuracy         : {(tg == pg).mean():.3f}")
    w(f"family accuracy        : {(tf == pf).mean():.3f}")
    if sp_ok.mean() > 0.90:
        w("")
        w("!! LEAKAGE WARNING: accuracy above 0.90 on external data is not plausible")
        w("!! for a model that has not seen the collection. Check whether this dataset")
        w("!! is one of the 25 sources inside FOR-species20K before reporting it.")
    w("")
    w("for reference, the same model on its own in-distribution split: species 0.821,")
    w("genus 0.840, family 0.858; and on source-holdout folds: species 0.157,")
    w("genus 0.225.")
    w("")
    w("-" * 78)
    w("BY SPECIES")
    w("-" * 78)
    w(f"{'true':<22}{'n':>5}{'species':>9}{'genus':>8}{'family':>8}   most predicted")
    for s in sorted(set(true), key=lambda x: -int((true == x).sum())):
        sel = true == s
        top = pd.Series(pred[sel]).value_counts().head(2)
        tops = ", ".join(f"{k}({v})" for k, v in top.items())
        w(f"{s:<22}{int(sel.sum()):>5}{sp_ok[sel].mean():>9.3f}"
          f"{(tg[sel] == pg[sel]).mean():>8.3f}{(tf[sel] == pf[sel]).mean():>8.3f}   {tops}")
    w("")
    w("-" * 78)
    w("BY PLOT")
    w("-" * 78)
    w(f"{'plot':<8}{'n':>5}{'species':>9}{'genus':>8}{'family':>8}")
    for p in sorted(trees["plot"].unique()):
        sel = (trees["plot"] == p).to_numpy()
        w(f"{p:<8}{int(sel.sum()):>5}{sp_ok[sel].mean():>9.3f}"
          f"{(tg[sel] == pg[sel]).mean():>8.3f}{(tf[sel] == pf[sel]).mean():>8.3f}")

    text = "\n".join(lines)
    out = args.out or DEFAULT_OUT.with_name(f"ext_validation_{Path(args.trees).stem}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
