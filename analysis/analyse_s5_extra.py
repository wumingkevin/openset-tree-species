# -*- coding: utf-8 -*-
"""跨属 / 跨采集活动的属内留出复现分析。

覆盖 Pinus(S5 原设定)、Acer、Picea 三个属的站点隔离版，以及 Quercus 的多来源版
（含"站点重叠 / 站点隔离"分层），用于回答"是否只是一个林区的特例"。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
import osr as O  # noqa: E402
from metrics import family_of, genus_of  # noqa: E402

ROOT = Path(".")
OUT = Path("results/tables_v2/s5_extra")
OUT.mkdir(parents=True, exist_ok=True)

GEN = [
    ("Pinus", "splits_clean/s5_pinus.json", "runs_v2", 3),
    ("Acer", "splits_clean/s5_acer.json", "runs_s5x", 3),
    ("Picea", "splits_clean/s5_picea.json", "runs_s5x", 3),
    ("Quercus", "splits_clean/s5_quercus.json", "runs_s5x", 3),
]
DET = ["msp", "energy", "mahalanobis", "knn"]


def run_dir(base, stem, f):
    return Path(base) / f"{stem}_f{f}"


def analyse(stem, split_path, base, nfolds):
    sp = json.loads(Path(split_path).read_text(encoding="utf-8"))
    unk = set(sp["unknown_treeIDs"])
    rows, strat = [], []
    for f in range(nfolds):
        d = run_dir(base, stem, f)
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
        train_sites = set(mt.loc[tr, "dataset"].astype(str))
        mdl = O.fit_mahalanobis(ft[tr], y[tr], shrink=1e-3)
        sc = {"msp": O.score_msp(lg), "energy": O.score_energy(lg, 1.0),
              "mahalanobis": O.score_mahalanobis(ft, mdl),
              "knn": O.score_knn(ft, ft[tr], k=10, seed=0)}
        pred = np.array(cl)[lg.argmax(1)]
        gt = genus_of(mt.species.astype(str).tolist())
        gp = genus_of(pred.tolist())
        fa = family_of(mt.species.astype(str).tolist())
        fp = family_of(pred.tolist())
        # 未知树是否与训练来源重叠
        site_shared = is_u & mt.dataset.astype(str).isin(train_sites).to_numpy()
        site_disj = is_u & ~mt.dataset.astype(str).isin(train_sites).to_numpy()
        for k in DET:
            thr = float(np.quantile(sc[k][is_v], 0.05))
            acc = is_u & (sc[k] >= thr)
            rej = is_u & (sc[k] < thr)
            rows.append(dict(genus=stem, fold=f, det=k, thr=thr,
                             n_unk=int(is_u.sum()),
                             n_acc=int(acc.sum()), n_rej=int(rej.sum()),
                             gen_acc=float((gt[acc] == gp[acc]).mean()) if acc.sum() else np.nan,
                             gen_rej=float((gt[rej] == gp[rej]).mean()) if rej.sum() else np.nan,
                             fam_acc=float((fa[acc] == fp[acc]).mean()) if acc.sum() else np.nan,
                             fam_rej=float((fa[rej] == fp[rej]).mean()) if rej.sum() else np.nan))
            for tag, m in (("site_shared", site_shared), ("site_disjoint", site_disj)):
                if m.sum() == 0:
                    continue
                a2 = acc & m
                r2 = rej & m
                strat.append(dict(genus=stem, fold=f, det=k, stratum=tag,
                                  n=int(m.sum()),
                                  n_acc=int(a2.sum()), n_rej=int(r2.sum()),
                                  gen_acc=float((gt[a2] == gp[a2]).mean()) if a2.sum() else np.nan,
                                  gen_rej=float((gt[r2] == gp[r2]).mean()) if r2.sum() else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(strat)


# 未知集的属/种/来源构成
meta = pd.read_csv("data/metadata/tree_metadata_dev.csv")
comp = []
for name, sp_path, _, _ in GEN:
    sp = json.loads(Path(sp_path).read_text(encoding="utf-8"))
    unk = set(sp["unknown_treeIDs"])
    u = meta[meta.treeID.isin(unk)]
    comp.append(dict(genus=name, n_unknown=len(u), n_species=u.species.nunique(),
                     n_campaigns=u.dataset.nunique(),
                     species=", ".join(sorted(u.species.unique()))))
comp = pd.DataFrame(comp)
comp.to_csv(OUT / "composition.csv", index=False)
print("=== 未知集构成 ===")
print(comp.to_string(index=False))

allr, alls = [], []
for name, sp_path, base, nf in GEN:
    stem = f"s5_{name.lower()}"
    r, s = analyse(stem, sp_path, base, nf)
    if len(r):
        allr.append(r)
    if len(s):
        alls.append(s)
    print(f"[{name}] 折数 {r.fold.nunique() if len(r) else 0}")
R = pd.concat(allr, ignore_index=True) if allr else pd.DataFrame()
S = pd.concat(alls, ignore_index=True) if alls else pd.DataFrame()
R.to_csv(OUT / "by_genus.csv", index=False)
S.to_csv(OUT / "by_stratum.csv", index=False)

if len(R):
    print("\n=== 按属汇总（三折均值；属召回 拒识 / 接受）===")
    piv = R.groupby(["genus", "det"]).agg(
        n_unk=("n_unk", "mean"), n_rej=("n_rej", "mean"),
        gen_acc=("gen_acc", "mean"), gen_rej=("gen_rej", "mean"),
        fam_acc=("fam_acc", "mean"), fam_rej=("fam_rej", "mean")).round(3)
    print(piv.to_string())
if len(S):
    print("\n=== Quercus 按站点关系分层（属召回 拒识 / 接受）===")
    q = S[S.genus == "s5_quercus"].groupby(["stratum", "det"]).agg(
        n=("n", "mean"), n_rej=("n_rej", "mean"),
        gen_acc=("gen_acc", "mean"), gen_rej=("gen_rej", "mean")).round(3)
    print(q.to_string())
print(f"\n[写出] {OUT}")
