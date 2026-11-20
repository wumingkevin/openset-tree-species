# -*- coding: utf-8 -*-
"""配对验证第一轮：红栎（Quercus rubra @ Seidl_2021）的 A/B 配对划分生成。

设计（与《面向 MEE 的配对验证与实验决策方案》一致）
- 测试树：Seidl_2021 的 58 棵 Q. rubra，永不进入训练/验证。
- 共同底集：全部来源排除 Seidl_2021、排除所有 Q. rubra，并隔离候选供体来源的全部 Q. robur。
- A 组：底集 + Seidl_2021 的 14 棵 Q. robur（同源供体）。
- B 组：底集 + 14 棵 Q. robur，来自另一来源（按树高与 A 组匹配后选取）。
- 两组共用底集、类别表、每类树数、渲染、骨干、轮数；只有供体来源不同。
"""
import json
import os

import numpy as np
import pandas as pd

OUT = "splits_paired"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
TARGET_SPECIES = "Quercus_rubra"
TEST_SOURCE = "Seidl_2021"
DONOR_SPECIES = "Quercus_robur"
DONOR_SOURCES_B = ["Calders_2022", "Frey_2022", "VanDeBerge_2021"]   # 逐个做敏感性
N_DONOR = 14

m = pd.read_csv("data/views_fix/meta.csv")
m = m[m.ok.astype(bool)].reset_index(drop=True)      # 位置 = views.npy 的行号
pos_of = {tid: i for i, tid in enumerate(m.treeID)}
meta = pd.read_csv("data/metadata/tree_metadata_dev.csv")
meta = meta.set_index("treeID")

test = m[(m.species == TARGET_SPECIES) & (m.dataset == TEST_SOURCE)].treeID.tolist()
assert len(test) == 58, len(test)
donor_a = m[(m.species == DONOR_SPECIES) & (m.dataset == TEST_SOURCE)].treeID.tolist()
assert len(donor_a) == 14, len(donor_a)

# 需要从底集排除的树：Seidl 全部、所有 Q. rubra、候选供体来源的全部 Q. robur
excl = set(m[m.dataset == TEST_SOURCE].treeID)
excl |= set(m[m.species == TARGET_SPECIES].treeID)
for s in DONOR_SOURCES_B:
    excl |= set(m[(m.species == DONOR_SPECIES) & (m.dataset == s)].treeID)
base = m[~m.treeID.isin(excl)].copy()
print(f"底集 {len(base)} 棵；排除 {len(excl)} 棵；类别 {base.species.nunique()}")

# 供体高度匹配：把 A 组 14 棵的树高作为目标分布，从候选来源里挑 14 棵
h_a = meta.loc[donor_a, "tree_H"].to_numpy()


def pick_matched(pool_ids, k):
    """按树高与 A 组分布贪心配对（不放回），返回选中的 k 棵。"""
    h = meta.loc[pool_ids, "tree_H"].to_numpy()
    order = np.argsort(h_a)                    # 目标由矮到高
    used = np.zeros(len(pool_ids), bool)
    chosen = []
    for tgt_idx in order:
        tgt = h_a[tgt_idx]
        cand = np.where(used, np.inf, np.abs(h - tgt))
        j = int(np.argmin(cand))
        used[j] = True
        chosen.append(pool_ids[j])
    return chosen[:k]


def strat_split(df, seed, val_frac=0.2):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for sp, g in df.groupby("species"):
        ids = g.treeID.to_numpy().copy()
        rng.shuffle(ids)
        n_va = max(1, int(round(len(ids) * val_frac)))
        va += ids[:n_va].tolist()
        tr += ids[n_va:].tolist()
    return tr, va


manifest = {"test_ids": test, "donor_A": donor_a, "donor_B": {}, "runs": []}
for src in DONOR_SOURCES_B:
    pool = m[(m.species == DONOR_SPECIES) & (m.dataset == src)].treeID.tolist()
    manifest["donor_B"][src] = pick_matched(pool, N_DONOR)
    print(f"  B 组候选 {src}: 池 {len(pool)} → 选 {len(manifest['donor_B'][src])}")

for seed in SEEDS:
    tr_ids, va_ids = strat_split(base, seed)
    for grp, donors in [("A", donor_a),
                        *[("B_" + k, v) for k, v in manifest["donor_B"].items()]]:
        train = tr_ids + list(donors)
        bad = set(train) & set(test)
        assert not bad, bad
        classes = sorted(set(m[m.treeID.isin(train)].species))
        rec = dict(seed=seed, group=grp, n_train=len(train), n_val=len(va_ids),
                   n_classes=len(classes), donor_source=(TEST_SOURCE if grp == "A" else grp[2:]),
                   n_donor=len(donors))
        manifest["runs"].append(rec)
        split = {
            "meta": {"design": "paired_congener_source", "target_species": TARGET_SPECIES,
                     "test_source": TEST_SOURCE, "group": grp, "seed": seed,
                     "donor_source": rec["donor_source"], "n_donor": len(donors)},
            "unknown_treeIDs": test,
            "folds": [{"fold": 0,
                       "train": [pos_of[t] for t in train],
                       "val": [pos_of[t] for t in va_ids],
                       "val_datasets": sorted(set(m[m.treeID.isin(va_ids)].dataset))}],
        }
        name = f"splits_paired/rubra_{grp}_seed{seed}.json"
        with open(name, "w", encoding="utf-8") as f:
            json.dump(split, f)

print("\n=== 冻结的划分清单 ===")
man = pd.DataFrame(manifest["runs"])
print(man.to_string(index=False))
with open(f"{OUT}/manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=1)

# 关键校验
print("\n=== 设计校验 ===")
chk = man.groupby("seed").agg(classes=("n_classes", "nunique"),
                             ntrain=("n_train", "nunique"))
print(chk.to_string())
print("  同一 seed 下 A/B 的类别数与训练树数一致：",
      bool((chk.classes == 1).all() and (chk.ntrain == 1).all()))
print(f"  测试树 {len(test)} 棵，均不在任何训练集内：",
      all(not (set(json.load(open(f"splits_paired/{r}.json", encoding='utf-8'))["folds"][0]["train"])
              & set(test))
          for r in [f"rubra_{g}_seed{s}" for s in SEEDS for g in ["A"] + ["B_" + x for x in DONOR_SOURCES_B]]))
print(f"  所有 Q. rubra 均已排除：",
      int(m[m.species == TARGET_SPECIES].treeID.isin(base.treeID).sum()) == 0)
