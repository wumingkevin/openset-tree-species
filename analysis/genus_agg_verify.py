# -*- coding: utf-8 -*-
"""验证属内概率聚合：在同一批被拒识树上比较 物种argmax映射 vs 属内概率聚合。"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import genus_of  # noqa: E402

OUT = "results/tables_v2/genus_agg"
os.makedirs(OUT, exist_ok=True)

S5 = {"Pinus": "splits_clean/s5_pinus.json", "Acer": "splits_clean/s5_acer.json",
      "Picea": "splits_clean/s5_picea.json", "Quercus": "splits_clean/s5_quercus.json"}
ALL_S5 = set().union(*[set(json.loads(open(v, encoding="utf-8").read())["unknown_treeIDs"])
                       for v in S5.values()])

rec = []
for b in range(5):
    run = f"runs_v2/S3_block{b}_f0"
    lg = np.load(f"{run}/logits.npy").astype(np.float64)
    ft = np.load(f"{run}/feats.npy").astype(np.float64)
    mt = pd.read_csv(f"{run}/meta.csv").astype({"dataset": str})
    cl = json.loads(open(f"{run}/classes.json", encoding="utf-8").read())
    sp = json.loads(open(f"splits/S3_block{b}.json", encoding="utf-8").read())
    c2i = {c: i for i, c in enumerate(cl)}
    y = np.array([c2i.get(s, -1) for s in mt.species])
    fd = sp["folds"][0]
    is_v = mt.treeID.isin(set(fd["val"])).to_numpy() & (y >= 0)
    is_t = mt.treeID.isin(set(fd["train"])).to_numpy()
    is_u = mt.treeID.isin(set(sp["unknown_treeIDs"])).to_numpy()
    tr = is_t & (y >= 0)

    cl_genera = sorted(set(genus_of(cl).tolist()))
    gi = {g: i for i, g in enumerate(cl_genera)}
    cl_g = genus_of(cl)
    tg = genus_of(mt.species.astype(str).tolist())

    # 目标未知树：物种无支持、属有支持、且在 S5 未知集中
    target = is_u & (y < 0) & np.isin(tg, cl_genera) & mt.treeID.isin(ALL_S5).to_numpy()
    # 同来源的已知树（用于评估回退代价）
    srcs = set(mt.loc[target, "dataset"])
    known_same_src = is_v & mt.dataset.isin(srcs).to_numpy()

    # softmax 概率与属内聚合
    z = lg - lg.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    agg = np.zeros((p.shape[0], len(cl_genera)))
    for j, g in enumerate(cl_g):
        agg[:, gi[g]] += p[:, j]
    pred_map = cl_g[[int(i) for i in lg.argmax(1)]]          # 方法 A
    pred_agg = np.array(cl_genera)[agg.argmax(1)]            # 方法 B
    # 基线：训练集最常见属
    tr_g, tr_cnt = np.unique(tg[tr], return_counts=True)
    const_genus = tr_g[tr_cnt.argmax()]

    mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
    sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
          "mahalanobis": O.score_mahalanobis(ft, mdl),
          "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}

    for name, s in sc.items():
        thr = float(np.quantile(s[is_v], 0.05))
        rej_u = target & (s < thr)
        rej_k = known_same_src & (s < thr)
        rec.append(dict(
            block=b, det=name,
            n_rej_unk=int(rej_u.sum()),
            ok_map=int((pred_map[rej_u] == tg[rej_u]).sum()),
            ok_agg=int((pred_agg[rej_u] == tg[rej_u]).sum()),
            ok_const=int((const_genus == tg[rej_u]).sum()),
            n_rej_known=int(rej_k.sum()),
            ok_known_map=int((pred_map[rej_k] == tg[rej_k]).sum()),
            ok_known_agg=int((pred_agg[rej_k] == tg[rej_k]).sum())))

d = pd.DataFrame(rec)
d.to_csv(f"{OUT}/by_block.csv", index=False)

print("=== 逐块 ===")
print(d.to_string(index=False))
print("\n=== 合计（五个 S3 块合并）===")
tot = d.drop(columns=["block"]).groupby("det").sum()
tot["recall_map"] = tot.ok_map / tot.n_rej_unk
tot["recall_agg"] = tot.ok_agg / tot.n_rej_unk
tot["净增"] = tot.ok_agg - tot.ok_map
print(tot[["n_rej_unk", "ok_map", "ok_agg", "净增", "recall_map", "recall_agg",
           "ok_const"]].round(3).to_string())
print("\n=== 已知树代价（同来源、被该门控拒识）===")
k = tot[["n_rej_known", "ok_known_map", "ok_known_agg"]].copy()
k["变化"] = k.ok_known_agg - k.ok_known_map
print(k.to_string())
print(f"\n[saved] {OUT}/by_block.csv")
