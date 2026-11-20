#!/usr/bin/env python3
"""Graded scanner-change probe: how much of the external drop is density and noise?

The Belgian collection sits 0.39 below the in-distribution score, but an external
collection changes country, forest, scanner and season at once, so the drop cannot be
attributed to any one of them. This probe varies one physical property at a time on
trees the model already knows: point density and ranging noise. The result is a curve
rather than a single cliff, and it bounds how much of the external drop an instrument
difference alone can explain.

The model is not retrained; only the test-side renders are degraded, which is the
deployment case. Each cloud is read once and reused across all degradation levels.

    python code/simulate_scanner.py --limit 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_views_fix import index_files, render_one, SCALE
from train_views import MVConvNeXt

OUT = Path("results/tables_fix/scanner_probe_fix.txt")   # 由 --out 覆盖
CH_MEAN = torch.tensor([7.4, 5.1, 8.6]).view(1, 3, 1, 1) / 255.0
CH_STD = torch.tensor([38.4, 28.2, 44.0]).view(1, 3, 1, 1) / 255.0

LEVELS = [
    ("100% points, no noise", 1.00, 0.00),
    ("50% points, no noise", 0.50, 0.00),
    ("25% points, no noise", 0.25, 0.00),
    ("10% points, no noise", 0.10, 0.00),
    ("100% points, 1 cm noise", 1.00, 0.01),
    ("100% points, 3 cm noise", 1.00, 0.03),
    ("25% points, 3 cm noise", 0.25, 0.03),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=Path, default=Path("splits/S1_seed0.json"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", type=Path, default=Path("runs/REF_all33"))
    ap.add_argument("--root", type=Path, default=Path("data/dev"))
    ap.add_argument("--meta", type=Path, default=Path("data/metadata/tree_metadata_dev.csv"))
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--out", type=Path, default=Path("results/tables_fix/scanner_probe_fix.txt"))
    args = ap.parse_args()
    global OUT
    OUT = args.out

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cls = json.loads((args.model / "classes.json").read_text(encoding="utf-8"))
    c2i = {c: i for i, c in enumerate(cls)}
    ck = torch.load(args.model / "best.pt", map_location=dev, weights_only=False)
    model = MVConvNeXt(len(cls), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    val = set(split["folds"][args.fold]["val"])
    meta = pd.read_csv(args.meta)
    meta = meta[meta.treeID.isin(val) & meta.species.isin(c2i)]
    if args.limit and args.limit < len(meta):
        # a head() slice is not random here: treeIDs are blocked by dataset, so the first
        # N trees are systematically easier and inflate the baseline
        meta = meta.sample(n=args.limit, random_state=0).sort_values("treeID")
    print(f"[test] {len(meta)} held-out trees, {len(LEVELS)} conditions", flush=True)

    index = index_files(args.root)
    views = [(360.0 * i / 4, 20.0) for i in range(4)]
    tally = {lab: [0, 0] for lab, _, _ in LEVELS}

    for n_done, (_, r) in enumerate(meta.iterrows(), 1):
        p = index.get(Path(str(r["filename"])).stem)
        if p is None:
            continue
        try:
            xyz0 = np.asarray(laspy.read(str(p)).xyz, dtype=np.float64)
        except Exception:  # noqa: BLE001
            continue
        if len(xyz0) < 32:
            continue
        seed = int(r["treeID"]) % 2**31
        for lab, keep, noise in LEVELS:
            xyz = xyz0
            n = max(32, int(len(xyz) * keep))
            if n < len(xyz):
                rng = np.random.default_rng(seed)
                xyz = xyz[rng.choice(len(xyz), n, replace=False)]
            if noise > 0:
                rng = np.random.default_rng(seed + 1)
                xyz = xyz + rng.normal(0, noise, xyz.shape)
            c = xyz - np.array([xyz[:, 0].mean(), xyz[:, 1].mean(),
                                (xyz[:, 2].min() + xyz[:, 2].max()) / 2.0])
            c = c / SCALE
            imgs = render_one(c, views, 192)
            x = torch.from_numpy(imgs).float().unsqueeze(0) / 255.0
            x = (x - CH_MEAN) / CH_STD
            with torch.no_grad():
                z, _ = model(x.to(dev))
            pred = cls[int(z.argmax(1))]
            tally[lab][0] += int(pred == r["species"])
            tally[lab][1] += 1
        if n_done % 100 == 0:
            print(f"   {n_done}/{len(meta)}", flush=True)

    lines = ["=" * 68, "GRADED SCANNER PROBE  (test-side degradation only)", "=" * 68,
             f"model {args.model}  |  {args.split} fold {args.fold}", ""]
    lines.append(f"{'condition':<28}{'n':>6}{'species acc':>14}")
    for lab, _, _ in LEVELS:
        c, s = tally[lab]
        lines.append(f"{lab:<28}{s:>6}{(c / s if s else float('nan')):>14.3f}")
    lines.append("")
    lines.append("references: in-distribution 0.821 | Belgium external 0.439 | S3 mean 0.157")
    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print()
    print(text)
    print(f"\n[wrote] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

