# -*- coding: utf-8 -*-
"""栎属未知集按"来源是否与同属训练树共享"分层重算。"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

GEN = [("Pinus", "splits_clean/s5_pinus.json", "runs_v2"),
       ("Acer", "splits_clean/s5_acer.json", "runs_s5x"),
       ("Picea", "splits_clean/s5_picea.json", "runs_s5x"),
       ("Quercus", "splits_clean/s5_quercus.json", "runs_s5x")]
rows = []

for name, sp_path, base in GEN:
    sp = json.loads(open(sp_path, encoding="utf-8").read())
    unk = set(sp["unknown_treeIDs"])
    stem = f"s5_{name.lower()}"
    for f in range(3):
        d = f"{base}/{stem}_f{f}"
        lg = np.load(f"{d}/logits.npy").astype(np.float64)
        ft = np.load(f"{d}/feats.npy").astype(np.float64)
        mt = pd.read_csv(f"{d}/meta.csv").astype({"dataset": str})
        cl = json.loads(open(f"{d}/classes.json", encoding="utf-8").read())
        c2i = {c: i for i, c in enumerate(cl)}
        y = np.array([c2i.get(s, -1) for s in mt.species])
        fd = sp["folds"][f]
        is_v = mt.treeID.isin(set(fd["val"])).to_numpy() & (y >= 0)
        is_t = mt.treeID.isin(set(fd["train"])).to_numpy()
        is_u = mt.treeID.isin(unk).to_numpy()
        tr = is_t & (y >= 0)
        # 同属已知物种在训练中使用的来源
        gen = name
        same_genus_tr = tr & mt.species.astype(str).str.startswith(gen + "_").to_numpy()
        congener_sites = set(mt.loc[same_genus_tr, "dataset"])
        mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
        sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
              "mahalanobis": O.score_mahalanobis(ft, mdl),
              "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}
        pred = np.array(cl)[lg.argmax(1)]
        gt = genus_of(mt.species.astype(str).tolist())
        gp = genus_of(pred.tolist())
        shared = is_u & mt.dataset.isin(congener_sites).to_numpy()
        alone = is_u & ~mt.dataset.isin(congener_sites).to_numpy()
        for k in ("msp", "energy", "mahalanobis", "knn"):
            thr = float(np.quantile(sc[k][is_v], 0.05))
            for tag, m in (("all", is_u), ("shared", shared), ("alone", alone)):
                if m.sum() == 0:
                    continue
                acc = m & (sc[k] >= thr)
                rej = m & (sc[k] < thr)
                rows.append(dict(genus=name, fold=f, det=k, stratum=tag, n=int(m.sum()),
                                 n_acc=int(acc.sum()), n_rej=int(rej.sum()),
                                 gen_acc=float((gt[acc] == gp[acc]).mean()) if acc.sum() else np.nan,
                                 gen_rej=float((gt[rej] == gp[rej]).mean()) if rej.sum() else np.nan))

d = pd.DataFrame(rows)
d.to_csv("results/tables_v2/s5_extra/by_same_genus_site.csv", index=False)

print("=== 按『来源是否与同属训练树共享』分层（MSP；属召回 接受 / 拒识）===")
q = d[d.det == "msp"].groupby(["genus", "stratum"]).agg(
    n=("n", "mean"), na=("gen_acc", "mean"), nr=("gen_rej", "mean")).round(3)
print(q.to_string())
print()
print("=== 四检测器在 95% 点的范围（供正文引用）===")
for g in ["Pinus", "Acer", "Picea", "Quercus"]:
    x = d[(d.genus == g) & (d.stratum == "all")]
    xs = d[(d.genus == g) & (d.stratum == "shared")]
    xa = d[(d.genus == g) & (d.stratum == "alone")]
    print(f"{g:<8} 全部: 接受 {x.gen_acc.min():.2f}-{x.gen_acc.max():.2f} 拒识 {x.gen_rej.min():.2f}-{x.gen_rej.max():.2f}"
          + (f" | 共享来源: {xs.gen_acc.min():.2f}-{xs.gen_acc.max():.2f} / {xs.gen_rej.min():.2f}-{xs.gen_rej.max():.2f}" if len(xs) else "")
          + (f" | 无同属: {xa.gen_acc.min():.2f}-{xa.gen_acc.max():.2f} / {xa.gen_rej.min():.2f}-{xa.gen_rej.max():.2f}" if len(xa) else ""))

# 去掉 Owen_2021 后的栎属
print()
print("=== 栎属去掉 Owen_2021（MSP 描述性，需另算）===")
