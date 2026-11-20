# -*- coding: utf-8 -*-
"""全来源留出下的属级回退：复用 S3 模型的零重训分析。

S3 把整组来源留出为未知来源组，因此这些树来自训练中从未出现的采集活动。
在其中挑出「物种未进训练、但属在训练中有代表」的树，即可检验：
模型遇到全新采集来源时，被拒识的未知树能否得到正确属标签。
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

OUT = "results/tables_v2/s3_allsource"
import os  # noqa: E402
os.makedirs(OUT, exist_ok=True)

TARGETS = {"p05": 0.05, "p10": 0.10, "p01": 0.01}
rows, comp = [], []

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

    # 类别表里的属
    cl_genera = set(genus_of(cl).tolist())
    tree_genus = genus_of(mt.species.astype(str).tolist())
    genus_supported = np.isin(tree_genus, list(cl_genera))
    # 目标：未知来源组 + 物种无训练支持 + 属有训练支持
    target = is_u & (y < 0) & genus_supported

    mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
    sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
          "mahalanobis": O.score_mahalanobis(ft, mdl),
          "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}
    pred = np.array(cl)[lg.argmax(1)]
    gt, gp = tree_genus, genus_of(pred.tolist())
    fa, fp = family_of(mt.species.astype(str).tolist()), family_of(pred.tolist())

    print(f"[block{b}] 目标树 {int(target.sum())}，验证已知 {int(is_v.sum())}")

    for name, s in sc.items():
        for tkey, q in TARGETS.items():
            thr = float(np.quantile(s[is_v], q))
            acc = target & (s >= thr)
            rej = target & (s < thr)
            for (gen, src), _ in mt[target].groupby(["species", "dataset"]):
                pass  # 下面按 genus 聚合
            sub = mt[target].copy()
            sub["g"] = tree_genus[target]
            for (gen, src), idx in sub.groupby(["g", "dataset"]).groups.items():
                m = mt.index.isin(idx)
                a2, r2 = acc & m, rej & m
                rows.append(dict(block=b, det=name, target=tkey, genus=gen, source=src,
                                 n=int(m.sum()), n_acc=int(a2.sum()), n_rej=int(r2.sum()),
                                 gen_acc=float((gt[a2] == gp[a2]).mean()) if a2.sum() else np.nan,
                                 gen_rej=float((gt[r2] == gp[r2]).mean()) if r2.sum() else np.nan,
                                 fam_acc=float((fa[a2] == fp[a2]).mean()) if a2.sum() else np.nan,
                                 fam_rej=float((fa[r2] == fp[r2]).mean()) if r2.sum() else np.nan))

d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/branch_by_genus_source.csv", index=False)

print("\n=== 按属×来源汇总（MSP，5% 操作点）===")
x = d[(d.det == "msp") & (d.target == "p05")].groupby(["genus", "source", "block"]).agg(
    n=("n", "first"), n_acc=("n_acc", "first"), n_rej=("n_rej", "first"),
    gen_acc=("gen_acc", "first"), gen_rej=("gen_rej", "first")).reset_index()
print(x.round(3).to_string(index=False))
print("\n=== 四个检测器合计（5% 操作点，按属×来源）===")
y2 = d[d.target == "p05"].groupby(["genus", "source"]).agg(
    n=("n", "first"),
    acc_min=("gen_acc", "min"), acc_max=("gen_acc", "max"),
    rej_min=("gen_rej", "min"), rej_max=("gen_rej", "max")).round(3)
print(y2.to_string())
print("\n=== 稳健性：不同操作点下拒识是否低于接受（MSP）===")
for t in ("p05", "p10", "p01"):
    z = d[(d.det == "msp") & (d.target == t)]
    ok = (z.gen_rej < z.gen_acc)
    print(f"  {t}: 逐组合 {int(ok.sum())}/{len(ok)} 拒识更低")
print(f"\n[saved] {OUT}/branch_by_genus_source.csv")
