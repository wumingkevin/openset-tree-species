#!/usr/bin/env python3
"""Generate the manuscript figures from the measured artefacts.

Section 4.9 described four figures but the document contained none. Three of the four can
be drawn from what the pipeline already produced; the fourth (usable rate against
rejection for a two-stage cascade) belonged to the cascade analysis that the measurements
made obsolete, so it is dropped rather than invented.

Figure 1  confusion matrix over the four genera that carry the error mass, species-level,
          with family boundaries drawn on the axes
Figure 2  open-set classification rate against false positive rate, four detectors,
          two panels (S1 random species, S5 within genus)
Figure 3  per species, the fraction of trees whose source is shared with another species
          of the same genus -- the quantity that makes a genus "signal" ambiguous

    python code/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("results/figures_v2")
GENERA = ["Carpinus", "Acer", "Fagus", "Tilia"]
FAMILY_OF_GENUS = {"Carpinus": "Betulaceae", "Acer": "Sapindaceae",
                   "Fagus": "Fagaceae", "Tilia": "Malvaceae"}


def load_run(path: Path):
    lg = np.load(path / "logits.npy")
    ft = np.load(path / "feats.npy")
    meta = pd.read_csv(path / "meta.csv")
    cls = json.loads((path / "classes.json").read_text(encoding="utf-8"))
    return lg, ft, meta, cls


def detector_scores(lg, ft, y, is_tr):
    """Delegate to osr.py's detectors so the figure cannot drift from the tables.

    An earlier version kept its own copies and they disagreed with the main scoring on all
    three feature-based detectors: it used the maximum logit as "energy" where the tables use
    the positive log-sum-exp, a single global covariance instead of per-class means with a
    pooled within-class covariance, and the mean of the k smallest squared distances instead
    of the k-th distance with an 8,000-feature reference library. A curve is only comparable
    to a table entry if both come from the same implementation, so the implementation lives in
    one place and this function calls it.
    """
    import sys
    sys.path.insert(0, str(Path("code").resolve()))
    import osr as O

    mdl = O.fit_mahalanobis(ft[is_tr], y[is_tr], shrink=O.SHRINK)
    return {
        "MSP": O.score_msp(lg),
        "Energy": O.score_energy(lg),
        "Mahalanobis": O.score_mahalanobis(ft, mdl),
        "kNN": O.score_knn(ft, ft[is_tr], k=10, seed=0),
    }


def fig1_confusion():
    lg, ft, meta, cls = load_run(Path("runs_v2/all_species_f0_f0"))
    split = json.loads(Path("splits_reference/all_species_f0.json").read_text(encoding="utf-8"))
    val = set(split["folds"][0]["val"])
    names = np.array(cls)
    c2i = {c: i for i, c in enumerate(cls)}
    y = np.array([c2i.get(s, -1) for s in meta["species"]])
    sel = meta.treeID.isin(val).to_numpy() & (y >= 0)
    true, pred = names[y[sel]], names[lg.argmax(1)[sel]]

    keep = np.array([t.split("_")[0] in GENERA for t in true])
    true, pred = true[keep], pred[keep]
    labels = sorted(set(true))
    idx = {s: i for i, s in enumerate(labels)}
    n = len(labels)
    # An extra column collects predictions that fall outside the displayed species. Without it
    # those rows were dropped before normalising, so the diagonal read as conditional accuracy
    # given a prediction inside the four genera rather than as true per-species recall.
    M = np.zeros((n, n + 1))
    for t, p in zip(true, pred):
        M[idx[t], idx.get(p, n)] += 1
    row = M.sum(1, keepdims=True)
    M = np.divide(M, row, out=np.zeros_like(M), where=row > 0)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    # pcolormesh 而非 imshow：PDF 里保持矢量，打印时格子边缘清晰
    im = ax.pcolormesh(np.arange(n + 2) - 0.5, np.arange(n + 1) - 0.5, M,
                       cmap="viridis", vmin=0, vmax=1, shading="flat")
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n + 1), labels + ["Other\npredicted\nspecies"], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted species", fontsize=9)
    ax.set_ylabel("true species", fontsize=9)
    for i in range(n):
        for j in range(n + 1):
            if M[i, j] > 0.02:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=7)
    # family separators
    fam = [FAMILY_OF_GENUS[s.split("_")[0]] for s in labels]
    bounds = [i - 0.5 for i in range(1, len(fam)) if fam[i] != fam[i - 1]]
    for b in bounds:
        ax.axhline(b, color="white", lw=2.0)
        ax.axvline(b, color="white", lw=2.0)
    ax.set_title("Fig. 1  Confusion over the four genera carrying most of the error\n"
                 "white lines separate families; rows normalised", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_confusion.png", dpi=300)
    fig.savefig(OUT / "fig1_confusion.pdf")
    plt.close(fig)
    print("[fig1] confusion matrix,", len(labels), "species")


def fig2_oscr():
    panels = [("S1 random species", Path("runs_v2/S1_seed0_f0"), "splits/S1_seed0.json"),
              ("S5 within genus", Path("runs_v2/s5_pinus_f0"), "splits_clean/s5_pinus.json")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, (title, run, split_path) in zip(axes, panels):
        lg, ft, meta, cls = load_run(run)
        split = json.loads(Path(split_path).read_text(encoding="utf-8"))
        val = set(split["folds"][0]["val"])
        unk = set(split["unknown_treeIDs"])
        c2i = {c: i for i, c in enumerate(cls)}
        y = np.array([c2i.get(s, -1) for s in meta["species"]])
        is_u = meta.treeID.isin(unk).to_numpy()
        is_val = meta.treeID.isin(val).to_numpy() & ~is_u
        is_tr = (y >= 0) & ~is_val & ~is_u
        sc = detector_scores(lg, ft, y, is_tr)
        for name, s in sc.items():
            sk, su = s[is_val], s[is_u]
            ok = (lg[is_val].argmax(1) == y[is_val])
            grid = np.quantile(np.concatenate([sk, su]), np.linspace(0, 1, 200))
            fpr, ccr = [], []
            for t in grid:
                # standard OSCR: correct AND accepted, over ALL knowns. Dividing by the
                # accepted count instead gives the conditional accuracy of the retained
                # set, which starts at 1 and is a different quantity.
                acc = sk >= t
                fpr.append(float((su >= t).mean()))
                ccr.append(float((ok & acc).sum() / len(ok)))
            ax.plot(fpr, ccr, lw=1.4, label=name)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("false positive rate (unknowns accepted)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("correct classification rate (OSCR)", fontsize=8)
    fig.suptitle("Fig. 2  Open-set classification rate (all knowns in the denominator)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_oscr.png", dpi=300)
    fig.savefig(OUT / "fig2_oscr.pdf")
    plt.close(fig)
    print("[fig2] OSCR curves for", len(panels), "partitions")


def fig3_source_sharing():
    m = pd.read_csv("data/metadata/tree_metadata_dev.csv")
    rows = []
    for sp, g in m.groupby("species"):
        genus = g.genus.iloc[0]
        same_genus = m[(m.genus == genus) & (m.species != sp)]
        shared = len(set(g.dataset) & set(same_genus.dataset))
        rows.append((sp, len(g), shared, len(set(g.dataset))))
    d = pd.DataFrame(rows, columns=["species", "n", "shared", "n_src"])
    d["frac"] = d.shared / d.n_src
    d = d.sort_values("frac", ascending=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    colors = ["#c0392b" if f >= 0.6 else "#e67e22" if f > 0 else "#7f8c8d"
              for f in d.frac]
    ax.barh(range(len(d)), d.frac, color=colors)
    ax.set_yticks(range(len(d)), [f"{s} ({n})" for s, n in zip(d.species, d.n)], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of the species' sources that also host a congener", fontsize=8)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color="black", ls=":", lw=1)
    ax.set_title("Fig. 3  How often a species shares its source with a congener\n"
                 "red: a same-genus error cannot be told apart from a site effect",
                 fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_source_sharing.png", dpi=300)
    fig.savefig(OUT / "fig3_source_sharing.pdf")
    plt.close(fig)
    print("[fig3] source sharing for", len(d), "species")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_confusion()
    fig2_oscr()
    fig3_source_sharing()
    print(f"\n[done] figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
