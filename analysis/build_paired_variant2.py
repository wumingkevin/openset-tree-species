# -*- coding: utf-8 -*-
"""变体 2（强操纵）：底集移除所有其他栎属，使属支持**只来自 14 棵供体**。

若变体 1（属支持充足）为 null，变体 2 检验：当同属训练数据只来自一个采集活动时，
换成另一活动的同属数据是否会显著改变对新来源目标物种的属级迁移。
"""
import json
import os

import numpy as np
import pandas as pd

OUT = "splits_paired"
SEEDS = [0, 1, 2]
TARGET, TEST_SRC = "Quercus_rubra", "Seidl_2021"
DONOR_SP = "Quercus_robur"
B_SRC = "Calders_2022"

m = pd.read_csv("data/views_fix/meta.csv")
m = m[m.ok.astype(bool)].reset_index(drop=True)
pos_of = {tid: i for i, tid in enumerate(m.treeID)}
meta = pd.read_csv("data/metadata/tree_metadata_dev.csv").set_index("treeID")
man = json.load(open(f"{OUT}/manifest.json", encoding="utf-8"))
test = man["test_ids"]
donor_a = man["donor_A"]
donor_b = man["donor_B"][B_SRC]

# 底集：排除 Seidl 全部、全部 Q. rubra、**所有栎属**（含供体来源的 robur）
excl = set(m[m.dataset == TEST_SRC].treeID)
excl |= set(m[m.species == TARGET].treeID)
excl |= set(m[m.species.astype(str).str.startswith("Quercus_")].treeID)
base = m[~m.treeID.isin(excl)].copy()
print(f"变体2 底集 {len(base)} 棵，类别 {base.species.nunique()}（不含任何栎属）")


def strat_split(df, seed, val_frac=0.2):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for sp, g in df.groupby("species"):
        ids = g.treeID.to_numpy().copy()
        rng.shuffle(ids)
        k = max(1, int(round(len(ids) * val_frac)))
        va += ids[:k].tolist()
        tr += ids[k:].tolist()
    return tr, va


for seed in SEEDS:
    tr_ids, va_ids = strat_split(base, seed)
    for grp, donors in (("A", donor_a), ("B_" + B_SRC, donor_b)):
        train = tr_ids + list(donors)
        classes = sorted(set(m[m.treeID.isin(train)].species))
        split = {
            "meta": {"design": "paired_variant2_only_donor_genus",
                     "target_species": TARGET, "test_source": TEST_SRC,
                     "group": grp, "seed": seed,
                     "donor_source": TEST_SRC if grp == "A" else B_SRC,
                     "n_donor": len(donors)},
            "unknown_treeIDs": test,
            "folds": [{"fold": 0,
                       "train": [pos_of[t] for t in train],
                       "val": [pos_of[t] for t in va_ids],
                       "val_datasets": sorted(set(m[m.treeID.isin(va_ids)].dataset))}],
        }
        name = f"{OUT}/v2_rubra_{grp}_seed{seed}.json"
        with open(name, "w", encoding="utf-8") as f:
            json.dump(split, f)
        print(f"  {name}: train {len(train)}，类 {len(classes)}，"
              f"唯一栎属类 {[c for c in classes if c.startswith('Quercus')]}")
