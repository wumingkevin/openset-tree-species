# -*- coding: utf-8 -*-
"""配对验证结果分析：58 棵测试树上的门控前属级正确率，A（同源供体）vs B（异源供体）。"""
import glob
import json
import os
import re
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, "code")
from metrics import genus_of  # noqa: E402

RUNS = "runs_paired"
OUT = "results/tables_v2/paired"
os.makedirs(OUT, exist_ok=True)
man = json.load(open("splits_paired/manifest.json", encoding="utf-8"))
test = list(man["test_ids"])
TEST_GENUS = "Quercus"

rows = []
for d in sorted(glob.glob(f"{RUNS}/*")):
    if not os.path.isdir(d) or not os.path.exists(f"{d}/logits.npy"):
        continue
    stem = os.path.basename(d)
    m = re.match(r"(v2_)?rubra_([AB])_?(.*)_seed(\d+)$", stem)
    if not m:
        continue
    variant = "v2" if m.group(1) else "v1"
    grp, donor, seed = m.group(2), (m.group(3) or "Seidl_2021"), int(m.group(4))
    lg = np.load(f"{d}/logits.npy")
    meta = pd.read_csv(f"{d}/meta.csv")
    cls = json.loads(open(f"{d}/classes.json", encoding="utf-8").read())
    pos = {tid: i for i, tid in enumerate(meta.treeID)}
    idx = [pos[t] for t in test if t in pos]
    pred = np.array(cls)[lg[idx].argmax(1)]
    pg = genus_of(pred.tolist())
    for tid, p, g in zip([t for t in test if t in pos], pred, pg):
        rows.append(dict(stem=stem, variant=variant, group=grp, donor=donor, seed=seed, treeID=tid,
                         pred_species=p, pred_genus=g, genus_ok=(g == TEST_GENUS)))
    print(f"[{stem}] 测试树 {len(idx)}，属正确 {int((pg == TEST_GENUS).sum())}"
          f"（{float((pg == TEST_GENUS).mean()):.3f}）")

d = pd.DataFrame(rows)
d.to_csv(f"{OUT}/per_tree.csv", index=False)
if not len(d):
    print("尚无完成的配对结果")
    raise SystemExit(0)

print("\n=== 逐次结果 ===")
t = d.groupby(["variant", "group", "donor", "seed"]).agg(
    n=("genus_ok", "size"), 属正确=("genus_ok", "sum"), 属正确率=("genus_ok", "mean"))
print(t.round(3).to_string())

print("\n=== A vs B 配对（同一变体、同一 seed、同一测试树）===")
pairs = []
for v in sorted(d.variant.unique()):
    dv = d[d.variant == v]
    A = dv[dv.group == "A"].set_index(["seed", "treeID"]).genus_ok
    for donor in sorted(dv[dv.group == "B"].donor.unique()):
        B = dv[(dv.group == "B") & (dv.donor == donor)].set_index(["seed", "treeID"]).genus_ok
        common = A.index.intersection(B.index)
        if not len(common):
            continue
        a, b = A.loc[common], B.loc[common]
        both = int((a & b).sum())
        only_a = int((a & ~b).sum())
        only_b = int((~a & b).sum())
        neither = int((~a & ~b).sum())
        per = []
        for s in sorted(dv.seed.unique()):
            idx = [i for i in common if i[0] == s]
            if idx:
                per.append((s, int(A.loc[idx].sum()), int(B.loc[idx].sum()), len(idx)))
        n = only_a + only_b
        p = (sum(comb(n, k) for k in range(only_a, n + 1)) / 2 ** n) * 2 if n else 1.0
        pairs.append(dict(变体=v, 供体来源=donor, n=len(common), 都对=both, 仅A对=only_a,
                          仅B对=only_b, 都错=neither,
                          A率=round(float(a.mean()), 3), B率=round(float(b.mean()), 3),
                          差=round(float(a.mean() - b.mean()), 3),
                          McNemar_p=round(min(p, 1.0), 3),
                          逐种子=" / ".join(f"s{s}:{x}-{y}" for s, x, y, _ in per)))
P = pd.DataFrame(pairs)
print(P.to_string(index=False) if len(P) else "（尚无 B 组结果）")
P.to_csv(f"{OUT}/paired_summary.csv", index=False)

print("\n=== 逐种子差值（A − B）===")
for donor in (P["供体来源"] if len(P) else []):
    sub = d[d.donor.isin([donor]) | (d.group == "A")]
    diffs = []
    for s in sorted(d.seed.unique()):
        a = sub[(sub.group == "A") & (sub.seed == s)].genus_ok.mean()
        b = sub[(sub.group == "B") & (sub.donor == donor) & (sub.seed == s)].genus_ok.mean()
        diffs.append(round(float(a - b), 3))
    print(f"  {donor}: {diffs}")
print(f"\n[saved] {OUT}")
