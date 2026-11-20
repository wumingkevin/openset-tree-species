#!/usr/bin/env python3
"""Rejection-branch fallback and the source-block acceptance split, done correctly.

Two bugs in the first version of this analysis were caught in review and are fixed here:

1. The threshold was calibrated on every row that was not in the unknown set, which in this
   pipeline includes the TRAINING trees (meta.csv carries train, validation and unknown rows
   together). The protocol calibrates on the held-out knowns of the fold only. Restricted to
   fold validation IDs here.

2. The energy score was returned as log(sum(exp(z))) after subtracting the row maximum, which is
   -log(MSP) and orders the trees backwards. The main script's score_energy adds the maximum
   back. This file now imports the detectors from osr.py instead of re-deriving them, so the two
   cannot drift apart again.

Coverage is also widened to all three S5 folds and all four detectors, as the review asked.

    python code/reject_branch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("code").resolve()))
import metrics as M  # noqa: E402
import osr as O  # noqa: E402

OUT = Path("results/tables_fix/reject_branch_fix.txt")   # --out 覆盖
RUNS = "runs_fix"    # --runs 覆盖
TRAIN_IDS: set[int] = set()


def scores_for(kind: str, run: Path):
    """Scores for one fold, using osr.py's own detector implementations."""
    lg = np.load(run / "logits.npy")
    meta = pd.read_csv(run / "meta.csv")
    cls = json.loads((run / "classes.json").read_text(encoding="utf-8"))
    c2i = {c: i for i, c in enumerate(cls)}
    y = np.array([c2i.get(s, -1) for s in meta.species])

    if kind == "msp":
        s = O.score_msp(lg)
    elif kind == "energy":
        s = O.score_energy(lg)
    else:
        feats = np.load(run / "feats.npy")
        train = (y >= 0) & meta.treeID.isin(TRAIN_IDS).to_numpy()
        if kind == "mahalanobis":
            mdl = O.fit_mahalanobis(feats[train], y[train], shrink=O.SHRINK)
            s = O.score_mahalanobis(feats, mdl)
        elif kind == "knn":
            s = O.score_knn(feats, feats[train], k=10)
        else:
            raise ValueError(kind)
    return s, y, meta


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_fix")
    ap.add_argument("--out", default="results/tables_fix/reject_branch_fix.txt")
    args, _ = ap.parse_known_args()
    global RUNS, OUT
    RUNS = args.runs
    OUT = Path(args.out)
    global TRAIN_IDS
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    lines = ["=" * 98,
             "A. SOURCE BLOCKS: acceptance split by whether the species has training support",
             "   threshold calibrated on that fold's VALIDATION knowns only (MSP)",
             "=" * 98]
    h = (f"{'block':<8}{'unknowns':>10}{'sp. seen':>10}{'accept(seen)':>14}"
         f"{'sp. unseen':>12}{'accept(unseen)':>16}{'gap':>9}")
    lines += [h, "-" * len(h)]

    for b in range(5):
        run = Path(RUNS) / f"S3_block{b}_f0"
        if not (run / "logits.npy").exists():
            continue
        sp = json.loads(Path(f"splits/S3_block{b}.json").read_text(encoding="utf-8"))
        fd = sp["folds"][0]
        unk, val = set(sp["unknown_treeIDs"]), set(fd["val"])
        TRAIN_IDS = set(fd["train"])

        s, y, meta = scores_for("msp", run)
        is_u = meta.treeID.isin(unk).to_numpy()
        is_v = meta.treeID.isin(val).to_numpy() & ~is_u & (y >= 0)
        thr = np.quantile(s[is_v], 0.05)
        acc = s >= thr

        seen = is_u & (y >= 0)
        unseen = is_u & (y < 0)
        a_s = acc[seen].mean() if seen.any() else float("nan")
        a_n = acc[unseen].mean() if unseen.any() else float("nan")
        lines.append(f"{b:<8}{int(is_u.sum()):>10}{int(seen.sum()):>10}{a_s:>14.3f}"
                     f"{int(unseen.sum()):>12}{a_n:>16.3f}{(a_s - a_n):>+9.3f}")

    lines += ["", "=" * 98,
              "B. FALLBACK ON THE BRANCH THE TEXT DESCRIBES (S5, all folds, all detectors)",
              "   thr = 5th percentile of that fold's validation knowns", "=" * 98]
    h2 = (f"{'detector':<13}{'fold':>5}{'set':<16}{'n':>6}{'sp ok':>7}{'genus ok':>10}"
          f"{'family ok':>11}")
    lines += [h2, "-" * len(h2)]

    sp5 = json.loads(Path("splits_clean/s5_pinus.json").read_text(encoding="utf-8"))
    unk5 = set(sp5["unknown_treeIDs"])
    agg: dict[tuple[str, str], list[tuple[int, int, int, int]]] = {}

    for fi, fd in enumerate(sp5["folds"]):
        run = Path(RUNS) / f"s5_pinus_f{fi}"
        if not (run / "logits.npy").exists():
            continue
        TRAIN_IDS = set(fd["train"])
        val = set(fd["val"])
        for det in ("msp", "energy", "mahalanobis", "knn"):
            s, y, meta = scores_for(det, run)
            is_u = meta.treeID.isin(unk5).to_numpy()
            is_v = meta.treeID.isin(val).to_numpy() & ~is_u & (y >= 0)
            thr = np.quantile(s[is_v], 0.05)
            acc = s >= thr
            cls = json.loads((run / "classes.json").read_text(encoding="utf-8"))
            pred = np.array(cls)[np.load(run / "logits.npy").argmax(1)]
            true = meta.species.to_numpy()
            for label, mask in (("rejected unk", is_u & ~acc),
                                ("accepted unk", is_u & acc),
                                ("rejected known", is_v & ~acc)):
                if mask.sum() == 0:
                    continue
                n = int(mask.sum())
                sp_ok = int((pred[mask] == true[mask]).sum())
                g_ok = int((M.genus_of(true[mask]) == M.genus_of(pred[mask])).sum())
                f_ok = int((M.family_of(true[mask]) == M.family_of(pred[mask])).sum())
                lines.append(f"{det:<13}{fi:>5}{label:<16}{n:>6}{sp_ok / n:>7.3f}"
                             f"{g_ok / n:>10.3f}{f_ok / n:>11.3f}")
                agg.setdefault((det, label), []).append((n, sp_ok, g_ok, f_ok))
            lines.append("")

    lines += ["=" * 98, "POOLED OVER THE THREE S5 FOLDS", "=" * 98]
    h3 = f"{'detector':<13}{'set':<16}{'n':>6}{'sp ok':>7}{'genus ok':>10}{'family ok':>11}"
    lines += [h3, "-" * len(h3)]
    for (det, label), vals in sorted(agg.items()):
        n = sum(v[0] for v in vals)
        lines.append(f"{det:<13}{label:<16}{n:>6}{sum(v[1] for v in vals) / n:>7.3f}"
                     f"{sum(v[2] for v in vals) / n:>10.3f}"
                     f"{sum(v[3] for v in vals) / n:>11.3f}")

    lines += ["",
              "'rejected known' rows show what a cascade would cost: how often the species the",
              "model would have emitted was already correct, which is the output a fallback has",
              "to beat rather than create."]
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


