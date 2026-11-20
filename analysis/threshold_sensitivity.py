# -*- coding: utf-8 -*-
"""阈值敏感性：S5 松属，已知保留目标 90/95/99% 下的接受/拒识分支召回。"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

sp = json.loads(open("splits_clean/s5_pinus.json", encoding="utf-8").read())
unk = set(sp["unknown_treeIDs"])
TARGETS = [0.90, 0.95, 0.99]

rows = []
for f in range(3):
    run = f"runs_v2/s5_pinus_f{f}"
    lg = np.load(f"{run}/logits.npy").astype(np.float64)
    ft = np.load(f"{run}/feats.npy").astype(np.float64)
    mt = pd.read_csv(f"{run}/meta.csv")
    cl = json.loads(open(f"{run}/classes.json", encoding="utf-8").read())
    c2i = {c: i for i, c in enumerate(cl)}
    y = np.array([c2i.get(s, -1) for s in mt.species])
    fd = sp["folds"][f]
    is_v = mt.treeID.isin(set(fd["val"])).to_numpy() & (y >= 0)
    is_t = mt.treeID.isin(set(fd["train"])).to_numpy()
    is_u = mt.treeID.isin(unk).to_numpy()
    tr_ok = is_t & (y >= 0)

    mdl = O.fit_mahalanobis(ft[tr_ok], y[tr_ok], shrink=1e-3)
    sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
          "mahalanobis": O.score_mahalanobis(ft, mdl),
          "knn": O.score_knn(ft, ft[tr_ok], k=10, seed=0)}
    pred = np.array(cl)[lg.argmax(1)]
    gt = genus_of(mt.species.astype(str).tolist()); gp = genus_of(pred.tolist())
    ftn = family_of(mt.species.astype(str).tolist()); fp = family_of(pred.tolist())

    for name, s in sc.items():
        for tgt in TARGETS:
            thr = float(np.quantile(s[is_v], 1 - tgt))
            acc = is_u & (s >= thr)
            rej = is_u & (s < thr)
            rows.append(dict(
                fold=f, detector=name, target=tgt, thr=thr,
                n_acc=int(acc.sum()), n_rej=int(rej.sum()),
                gen_acc=float((gt[acc] == gp[acc]).mean()) if acc.sum() else np.nan,
                gen_rej=float((gt[rej] == gp[rej]).mean()) if rej.sum() else np.nan,
                fam_acc=float((ftn[acc] == fp[acc]).mean()) if acc.sum() else np.nan,
                fam_rej=float((ftn[rej] == fp[rej]).mean()) if rej.sum() else np.nan,
                still_rejects=bool(rej.sum() > 0),
            ))
d = pd.DataFrame(rows)
d.to_csv("results/tables_v2/threshold_sensitivity_v2.csv", index=False)

print("=== 逐折结果（属召回；n 为该分支的未知树-模型记录数）===")
print(d.groupby(["detector", "target"]).agg(
    阈值=("thr", "mean"),
    接受n=("n_acc", "mean"),
    拒识n=("n_rej", "mean"),
    属召回_接受=("gen_acc", "mean"),
    属召回_拒识=("gen_rej", "mean"),
    科召回_接受=("fam_acc", "mean"),
    科召回_拒识=("fam_rej", "mean"),
).round(3).to_string())
print()
ok = d.groupby(["detector", "target"]).apply(
    lambda g: (g.gen_rej < g.gen_acc).all(), include_groups=False)
print("每一折都满足『拒识分支属召回 < 接受分支』：")
print(ok.to_string())
print()
print("[saved] results/tables_v2/threshold_sensitivity_v2.csv")
