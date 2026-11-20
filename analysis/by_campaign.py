# -*- coding: utf-8 -*-
"""按采集活动（dataset）拆分未知集，看拒识分支的属召回缺陷是否在多个活动上分别成立。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

GEN = [("Pinus", "splits_clean/s5_pinus.json", "runs_v2"),
       ("Acer", "splits_clean/s5_acer.json", "runs_s5x"),
       ("Picea", "splits_clean/s5_picea.json", "runs_s5x"),
       ("Quercus", "splits_clean/s5_quercus.json", "runs_s5x")]
OUT = Path("results/tables_v2/s5_extra")
rows = []

for name, sp_path, base in GEN:
    sp = json.loads(Path(sp_path).read_text(encoding="utf-8"))
    unk = set(sp["unknown_treeIDs"])
    stem = f"s5_{name.lower()}"
    for f in range(3):
        d = Path(base) / f"{stem}_f{f}"
        if not (d / "logits.npy").exists():
            continue
        lg = np.load(d / "logits.npy").astype(np.float64)
        ft = np.load(d / "feats.npy").astype(np.float64)
        mt = pd.read_csv(d / "meta.csv")
        cl = json.loads((d / "classes.json").read_text(encoding="utf-8"))
        c2i = {c: i for i, c in enumerate(cl)}
        y = np.array([c2i.get(s, -1) for s in mt.species])
        fd = sp["folds"][f]
        is_v = mt.treeID.isin(set(fd["val"])).to_numpy() & (y >= 0)
        is_t = mt.treeID.isin(set(fd["train"])).to_numpy()
        is_u = mt.treeID.isin(unk).to_numpy()
        tr = is_t & (y >= 0)
        mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
        sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
              "mahalanobis": O.score_mahalanobis(ft, mdl),
              "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}
        pred = np.array(cl)[lg.argmax(1)]
        gt = genus_of(mt.species.astype(str).tolist())
        gp = genus_of(pred.tolist())
        for k in ("msp", "mahalanobis"):
            thr = float(np.quantile(sc[k][is_v], 0.05))
            acc = is_u & (sc[k] >= thr)
            rej = is_u & (sc[k] < thr)
            for camp, sub in mt[is_u].groupby(mt.dataset.astype(str)):
                m = mt.index.isin(sub.index)
                a2, r2 = acc & m, rej & m
                if m.sum() < 10:
                    continue
                rows.append(dict(genus=name, fold=f, det=k, campaign=camp,
                                 n=int(m.sum()), n_acc=int(a2.sum()), n_rej=int(r2.sum()),
                                 gen_acc=float((gt[a2] == gp[a2]).mean()) if a2.sum() else np.nan,
                                 gen_rej=float((gt[r2] == gp[r2]).mean()) if r2.sum() else np.nan))

d = pd.DataFrame(rows)
d.to_csv(OUT / "by_campaign.csv", index=False)

print("=== 每个采集活动单独看（MSP；属召回 接受 → 拒识）===")
g = (d[d.det == "msp"].groupby(["genus", "campaign"])
     .agg(n=("n", "mean"), n_rej=("n_rej", "mean"),
          gen_acc=("gen_acc", "mean"), gen_rej=("gen_rej", "mean")).round(3))
print(g.to_string())
n_acc_worse = int((g.gen_rej < g.gen_acc).sum())
print(f"\n共 {len(g)} 个『属×活动』单元，其中拒识分支更低: {n_acc_worse}")
print("\n=== 只看未知树 ≥30 的单元 ===")
big = g[g.n >= 30]
print(big.to_string())
print(f"其中拒识更低: {int((big.gen_rej < big.gen_acc).sum())} / {len(big)}")
