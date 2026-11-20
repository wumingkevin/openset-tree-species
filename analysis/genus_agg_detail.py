# -*- coding: utf-8 -*-
"""补充核实：逐来源改对数、已知类树的代价、恒定属基线。"""
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

unk_rows, known_rows = [], []
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

    genera = sorted(set(genus_of(cl).tolist()))
    gi = {g: i for i, g in enumerate(genera)}
    cl_g = genus_of(cl)
    tg = genus_of(mt.species.astype(str).tolist())

    z = lg - lg.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    agg = np.zeros((p.shape[0], len(genera)))
    for j, g in enumerate(cl_g):
        agg[:, gi[g]] += p[:, j]
    pred_map = cl_g[[int(i) for i in lg.argmax(1)]]
    pred_agg = np.array(genera)[agg.argmax(1)]
    tr_g, cnt = np.unique(tg[tr], return_counts=True)
    const_genus = tr_g[cnt.argmax()]

    # 每属训练物种数（用于分层）
    n_sp = {}
    for g in genera:
        n_sp[g] = len({c for c, gg in zip(cl, cl_g) if gg == g})

    s = O.score_msp(lg)                        # 主文主检测器
    thr = float(np.quantile(s[is_v], 0.05))

    tgt_unk = is_u & (y < 0) & np.isin(tg, genera)
    tgt_known = is_u & (y >= 0)                # 未知来源组中、物种有训练支持的树

    for tag, mask in (("unknown", tgt_unk), ("known", tgt_known)):
        rej = mask & (s < thr)
        sub = mt[rej].copy()
        if not len(sub):
            continue
        sub["g_true"] = np.array(tg)[rej]
        sub["g_map"] = np.array(pred_map)[rej]
        sub["g_agg"] = np.array(pred_agg)[rej]
        sub["ok_map"] = sub.g_true == sub.g_map
        sub["ok_agg"] = sub.g_true == sub.g_agg
        sub["ok_const"] = const_genus == sub.g_true
        sub["n_sp"] = sub.g_true.map(n_sp).astype(int)
        sub["block"] = b
        (unk_rows if tag == "unknown" else known_rows).append(sub)

U = pd.concat(unk_rows, ignore_index=True)
K = pd.concat(known_rows, ignore_index=True)

print("=== 未知树：逐来源改对/改错（MSP，5%）===")
t = U.groupby(["g_true", "dataset"]).agg(
    n=("ok_map", "size"), map_=("ok_map", "sum"), agg=("ok_agg", "sum"),
    const=("ok_const", "sum"))
t["净增"] = t["agg"] - t["map_"]
print(t.to_string())
sub = U[U.dataset.isin(["Owen_2021", "Xi_2020a", "Xi_2020b"])]
print("\n方案提到的两组：")
for (g, ds), z2 in sub.groupby(["g_true", "dataset"]):
    print(f"  {g} {ds}: 拒识 {len(z2)}，改对 {int((z2.ok_agg & ~z2.ok_map).sum())}，"
          f"改错 {int((z2.ok_map & ~z2.ok_agg).sum())}，净 {int(z2.ok_agg.sum()-z2.ok_map.sum()):+d}")

print("\n=== 已知类树（同来源、被拒识）的代价 ===")
if len(K):
    k = K.groupby("dataset").agg(n=("ok_map", "size"), map_=("ok_map", "sum"),
                                 agg=("ok_agg", "sum"))
    k["改对"] = K[K.ok_agg & ~K.ok_map].groupby("dataset").size().reindex(k.index).fillna(0).astype(int)
    k["改错"] = K[K.ok_map & ~K.ok_agg].groupby("dataset").size().reindex(k.index).fillna(0).astype(int)
    print(k.to_string())
    print(f"合计: 拒识 {len(K)}，原方法正确 {int(K.ok_map.sum())}，聚合后 {int(K.ok_agg.sum())}，"
          f"改对 {int((K.ok_agg & ~K.ok_map).sum())}，改错 {int((K.ok_map & ~K.ok_agg).sum())}")

print("\n=== 按目标属的训练物种数分层（未知树）===")
print(U.groupby("n_sp").agg(n=("ok_map", "size"), map_=("ok_map", "sum"),
                            agg=("ok_agg", "sum"), const=("ok_const", "sum")).to_string())

print("\n=== 恒定属基线对比（未知树合计）===")
print(f"  拒识 {len(U)}：映射 {int(U.ok_map.sum())}，聚合 {int(U.ok_agg.sum())}，"
      f"恒定属 {int(U.ok_const.sum())}")
U.to_csv(f"{OUT}/unknown_by_source.csv", index=False)
if len(K):
    K.to_csv(f"{OUT}/known_cost.csv", index=False)
print(f"\n[saved] {OUT}")
