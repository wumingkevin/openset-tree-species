#!/usr/bin/env python3
"""Does the rejection gate close on a species the model has never seen?

The within-benchmark result is that the gate barely closes: at a threshold keeping 95% of
known trees, 65-97% of unseen trees are still accepted. That measurement always involved
species that still had congeners or confamilials in training. This script measures the
cleanest version of the question: an external collection of trees whose species
(Platanus x hispanica) has no representative anywhere in the label space.

Reported per detector: the acceptance rate at the threshold calibrated on the FOR
validation knowns, plus the distribution of predicted species. A model that knows when it
is out of its depth would accept close to 5% here.

    python code/ext_novel_reject.py
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

OUT = Path("results/tables_fix/ext_novel_reject_fix.txt")


def scores_all(logits: np.ndarray, feats: np.ndarray, logits_tr: np.ndarray,
               feats_tr: np.ndarray, k: int = 10):
    """Return dict of scores, higher = more likely known."""
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    p = softmax(logits)
    out = {"msp": p.max(1), "energy": logits.max(1)}

    # Mahalanobis against pooled covariance of the training features
    mu = feats_tr.mean(0)
    cov = np.cov(feats_tr, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-3 * np.trace(cov) / cov.shape[0]
    prec = np.linalg.pinv(cov)
    d = feats - mu
    out["mahalanobis"] = -np.einsum("ij,jk,ik->i", d, prec, d)

    # kNN distance to the training features, computed in row blocks: the full
    # pairwise tensor for 3.5k queries against 14k references is 143 GiB.
    tr = feats_tr
    tr_sq = (tr ** 2).sum(1)
    ks = np.empty(len(feats), dtype=np.float64)
    block = 256
    for s in range(0, len(feats), block):
        b = feats[s:s + block]
        d2 = (b ** 2).sum(1)[:, None] + tr_sq[None, :] - 2.0 * b @ tr.T
        ks[s:s + block] = np.partition(d2, k, axis=1)[:, :k].mean(1)
    out["knn"] = -ks
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("runs/REF_all33"))
    ap.add_argument("--views", type=Path, default=Path("data/_ext/stp_views/views.npy"))
    ap.add_argument("--meta", type=Path, default=Path("data/_ext/stp_views/meta.csv"))
    ap.add_argument("--split", type=Path,
                    default=Path("splits_reference/all_species_f0.json"))
    args = ap.parse_args()

    try:
        import sys2
    except Exception:
        pass
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

    # reference: training rows and validation-known rows from the FOR split
    ref_meta = pd.read_csv(args.model / "meta.csv")
    ref_lg = np.load(args.model / "logits.npy")
    ref_ft = np.load(args.model / "feats.npy")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    val_ids = set(split["folds"][0]["val"])

    c2i = {c: i for i, c in enumerate(cls)}
    ref_y = np.array([c2i.get(s, -1) for s in ref_meta.species])
    known = ref_y >= 0
    is_val = ref_meta.treeID.isin(val_ids).to_numpy() & known
    # training rows = known but not in val
    is_tr = known & ~is_val

    sc_ref = scores_all(ref_lg[is_val], ref_ft[is_val], ref_lg[is_tr], ref_ft[is_tr])

    # external trees
    vmeta = pd.read_csv(args.meta)
    ds = MVViews(args.views, vmeta, np.arange(len(vmeta)),
                 np.zeros(len(vmeta), dtype=np.int64))
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    lg, ft = [], []
    with torch.no_grad():
        for x, _ in dl:
            z, g = model(x.to(dev))
            lg.append(z.float().cpu().numpy())
            ft.append(g.float().cpu().numpy())
    lg, ft = np.concatenate(lg), np.concatenate(ft)
    sc_ext = scores_all(lg, ft, ref_lg[is_tr], ref_ft[is_tr])

    names = np.array(cls)
    pred = names[lg.argmax(1)]
    top = pd.Series(pred).value_counts().head(6)

    lines = []
    w = lines.append
    w("=" * 78)
    w("REJECTION ON A SPECIES ABSENT FROM THE LABEL SPACE")
    w("=" * 78)
    w(f"external trees          : {len(lg)}")
    w("species                 : Platanus x hispanica (London plane), never in training")
    w("threshold               : 5th percentile of the FOR validation knowns")
    w("expected acceptance if the gate worked : about 5%")
    w("")
    w(f"{'detector':<14}{'threshold':>12}{'accept rate':>14}")
    for sc in ("msp", "energy", "mahalanobis", "knn"):
        thr = float(np.quantile(sc_ref[sc], 0.05))
        acc = float((sc_ext[sc] >= thr).mean())
        w(f"{sc:<14}{thr:>12.3f}{acc*100:>13.1f}%")
    w("")
    w("most frequent predictions:")
    for k2, v in top.items():
        w(f"   {k2:<26}{v:>5}  ({v/len(pred)*100:.0f}%)")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

