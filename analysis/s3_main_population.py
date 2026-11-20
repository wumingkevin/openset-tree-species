# -*- coding: utf-8 -*-
"""主分析人群（与 S5 共有的未知树）+ 同一批树的 S3 / S5 模型对照。"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

OUT = "results/tables_v2/s3_allsource"
os.makedirs(OUT, exist_ok=True)

S5 = {"Pinus": ("splits_clean/s5_pinus.json", "runs_v2", "s5_pinus"),
      "Acer": ("splits_clean/s5_acer.json", "runs_s5x", "s5_acer"),
      "Picea": ("splits_clean/s5_picea.json", "runs_s5x", "s5_picea"),
      "Quercus": ("splits_clean/s5_quercus.json", "runs_s5x", "s5_quercus")}
S5_UNK = {}
for g, (sp_path, _, _) in S5.items():
    S5_UNK[g] = set(json.loads(open(sp_path, encoding="utf-8").read())["unknown_treeIDs"])
ALL_S5 = set().union(*S5_UNK.values())

rows = []
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
    cl_genera = list(set(genus_of(cl).tolist()))
    tg = genus_of(mt.species.astype(str).tolist())
    target = is_u & (y < 0) & np.isin(tg, cl_genera) & mt.treeID.isin(ALL_S5).to_numpy()

    mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
    sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
          "mahalanobis": O.score_mahalanobis(ft, mdl),
          "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}
    pred = np.array(cl)[lg.argmax(1)]
    gp = genus_of(pred.tolist())
    fp = family_of(pred.tolist())
    fa = family_of(mt.species.astype(str).tolist())

    for name, s in sc.items():
        thr = float(np.quantile(s[is_v], 0.05))
        acc = target & (s >= thr)
        rej = target & (s < thr)
        sub = mt[target].copy()
        sub["g"] = np.array(tg)[target]
        sub["in_acc"], sub["in_rej"] = acc[target], rej[target]
        sub["g_ok"] = (np.array(tg)[target] == gp[target])
        sub["f_ok"] = (fa[target] == fp[target])
        sub["block"] = b
        sub["det"] = name
        rows.append(sub[["treeID", "g", "dataset", "block", "det", "in_acc", "in_rej", "g_ok", "f_ok"]])

d = pd.concat(rows, ignore_index=True)
d.to_csv(f"{OUT}/per_tree_s3.csv", index=False)
print(f"S3 主分析人群：{d.treeID.nunique()} 棵不同树")
print(d.groupby(["g", "dataset"]).treeID.nunique().to_string())

m = d[d.det == "msp"].groupby(["g", "dataset"]).agg(
    n=("treeID", "nunique"),
    n_acc=("in_acc", "sum"), n_rej=("in_rej", "sum"),
    gen_acc=("g_ok", lambda s: s[d.loc[s.index, "in_acc"]].mean() if d.loc[s.index, "in_acc"].any() else np.nan),
    gen_rej=("g_ok", lambda s: s[d.loc[s.index, "in_rej"]].mean() if d.loc[s.index, "in_rej"].any() else np.nan))
print("\n=== 主分析人群（MSP，5%）===")
print(m.round(3).to_string())
m.to_csv(f"{OUT}/main_population.csv")

# ---- 同一批树的 S5 模型对照 ----
print("\n=== 同一批树的 S5 模型结果（MSP，5%）===")
rec = []
for g, (sp_path, base, stem) in S5.items():
    sp = json.loads(open(sp_path, encoding="utf-8").read())
    unk = S5_UNK[g]
    for f in range(3):
        run = f"{base}/{stem}_f{f}"
        if not os.path.exists(f"{run}/logits.npy"):
            continue
        lg = np.load(f"{run}/logits.npy").astype(np.float64)
        ft = np.load(f"{run}/feats.npy").astype(np.float64)
        mt = pd.read_csv(f"{run}/meta.csv").astype({"dataset": str})
        cl = json.loads(open(f"{run}/classes.json", encoding="utf-8").read())
        c2i = {c: i for i, c in enumerate(cl)}
        y = np.array([c2i.get(s, -1) for s in mt.species])
        fd = sp["folds"][f]
        is_v = mt.treeID.isin(set(fd["val"])).to_numpy() & (y >= 0)
        is_t = mt.treeID.isin(set(fd["train"])).to_numpy()
        is_u = mt.treeID.isin(unk).to_numpy()
        tr = is_t & (y >= 0)
        s = O.score_msp(lg)
        thr = float(np.quantile(s[is_v], 0.05))
        pred = np.array(cl)[lg.argmax(1)]
        tg = genus_of(mt.species.astype(str).tolist())
        gp = genus_of(pred.tolist())
        sub = mt[is_u].copy()
        sub["g"] = np.array(tg)[is_u]
        sub["in_acc"] = (s[is_u] >= thr)
        sub["g_ok"] = (np.array(tg)[is_u] == np.array(gp)[is_u])
        sub["fold"] = f
        rec.append(sub[["treeID", "g", "dataset", "in_acc", "g_ok", "fold"]])
s5d = pd.concat(rec, ignore_index=True)
s5d.to_csv(f"{OUT}/per_tree_s5.csv", index=False)
z = s5d.groupby(["g", "dataset"]).agg(
    n=("treeID", "nunique"), n_acc=("in_acc", "sum"),
    gen_acc=("g_ok", lambda s: s[s5d.loc[s.index, "in_acc"]].mean() if s5d.loc[s.index, "in_acc"].any() else np.nan),
    gen_rej=("g_ok", lambda s: s[~s5d.loc[s.index, "in_acc"]].mean() if (~s5d.loc[s.index, "in_acc"]).any() else np.nan))
print(z.round(3).to_string())
z.to_csv(f"{OUT}/s5_same_trees.csv")
print(f"\n[saved] {OUT}")
