#!/usr/bin/env python3
"""Open-set metrics, numpy only.

Convention used everywhere in this project
------------------------------------------
    score  : higher == more likely to belong to a KNOWN class
    known  : samples whose species is in the training label space
    unknown: samples whose species was never in training

Every metric here takes scores in that convention, so MSP, Energy, Mahalanobis and
kNN all need the same orientation (the scorers are responsible for the sign flip).

Metrics
-------
    auroc               known vs unknown separability
    oscr_auc            area under the open-set classification rate curve
    fpr_at_tpr          fraction of knowns wrongly rejected when 95% of unknowns are
                        caught -- the number a forester actually cares about
    oscr_at_fpr         CCR at a fixed FPR, i.e. "known accuracy at 5% unknown leakage"
    osa                 open-set accuracy (Bendale & Boult)
    macro_recall        unweighted mean per-species recall
    ece                 expected calibration error
    genus_recall        genus-level accuracy, optionally restricted to a mask
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- ranking
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared (equivalent to scipy.stats.rankdata)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i, n = 0, len(xs)
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive scores above a random negative). Ties count 0.5."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = _rankdata(np.concatenate([pos, neg]))
    r_pos = r[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def auroc_known_vs_unknown(known_scores, unknown_scores) -> float:
    return auroc(known_scores, unknown_scores)


# ------------------------------------------------------------------------ OSCR
def oscr_curve(known_scores, known_correct, unknown_scores, n_points: int = 400):
    """Return (fpr, ccr) sorted by increasing fpr.

    Sweeping the threshold from high to low: more unknowns are accepted (fpr rises)
    and more knowns are accepted (ccr rises, and it is capped by the closed-set
    accuracy because a wrong known can never add to ccr).
    """
    ks = np.asarray(known_scores, dtype=np.float64).ravel()
    us = np.asarray(unknown_scores, dtype=np.float64).ravel()
    kc = np.asarray(known_correct, dtype=bool).ravel()
    if len(ks) == 0 or len(us) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Candidate thresholds come from the observed scores themselves.  Quantile-only
    # sampling collapses to a single point when every unknown shares one score, which
    # made the curve degenerate (and the area NaN), so anchor both ends explicitly.
    pooled = np.concatenate([ks, us])
    cand = np.unique(pooled)
    if len(cand) > n_points:
        cand = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, n_points)))
    lo = float(pooled.min()) - 1.0
    hi = float(pooled.max()) + 1.0
    thr = np.concatenate([[hi], cand, [lo]])

    fpr = np.array([(us >= t).mean() for t in thr])
    ccr = np.array([(kc & (ks >= t)).sum() / len(ks) for t in thr])
    order = np.argsort(fpr, kind="mergesort")
    return fpr[order], ccr[order]


def oscr_auc(known_scores, known_correct, unknown_scores, n_points: int = 400) -> float:
    fpr, ccr = oscr_curve(known_scores, known_correct, unknown_scores, n_points)
    if len(fpr) < 2:
        return float("nan")
    return float(np.trapezoid(ccr, fpr))


def oscr_at_fpr(known_scores, known_correct, unknown_scores, target_fpr: float = 0.05) -> float:
    fpr, ccr = oscr_curve(known_scores, known_correct, unknown_scores)
    idx = np.searchsorted(fpr, target_fpr)
    idx = int(np.clip(idx, 0, len(ccr) - 1))
    return float(ccr[idx])


def fpr_at_tpr(known_scores, unknown_scores, tpr: float = 0.95) -> float:
    """`tpr` is recall on the UNKNOWN class: the fraction of unknowns correctly
    flagged as unknown. Report the collateral damage -- the fraction of knowns that
    fall below the same threshold and are wrongly rejected.

    thr = quantile(unknown, tpr): at tpr=0.95, 95% of unknowns sit below it.
    """
    ks = np.asarray(known_scores, dtype=np.float64).ravel()
    us = np.asarray(unknown_scores, dtype=np.float64).ravel()
    if len(ks) == 0 or len(us) == 0:
        return float("nan")
    thr = float(np.quantile(us, tpr))
    return float((ks < thr).mean())


def threshold_at_tpr(unknown_scores, tpr: float = 0.95) -> float:
    """Operating point that rejects `tpr` of the samples it is computed on.

    MUST be computed from synthetic pseudo-unknowns only, never from the real
    unknown set, or every number downstream is optimistic.
    """
    us = np.asarray(unknown_scores, dtype=np.float64).ravel()
    return float(np.quantile(us, tpr))


def osa(known_scores, known_correct, unknown_scores, thr: float) -> float:
    """Open-set accuracy at a fixed threshold: correct knowns kept + unknowns rejected."""
    ks = np.asarray(known_scores, dtype=np.float64).ravel()
    kc = np.asarray(known_correct, dtype=bool).ravel()
    us = np.asarray(unknown_scores, dtype=np.float64).ravel()
    if len(ks) + len(us) == 0:
        return float("nan")
    tp = int((kc & (ks >= thr)).sum())
    tn = int((us < thr).sum())
    return float((tp + tn) / (len(ks) + len(us)))


# -------------------------------------------------------------------- accuracy
def macro_recall(y_true, y_pred, labels=None) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(y_true)
    rec = []
    for c in labels:
        m = y_true == c
        if m.any():
            rec.append(float((y_pred[m] == c).mean()))
    return float(np.mean(rec)) if rec else float("nan")


def per_class_recall(y_true, y_pred, labels=None, names=None):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if labels is None:
        labels = np.unique(y_true)
    out = {}
    for i, c in enumerate(labels):
        m = y_true == c
        key = names[i] if names is not None else c
        out[key] = float((y_pred[m] == c).mean()) if m.any() else float("nan")
    return out


# --------------------------------------------------------------- calibration
def ece(probs: np.ndarray, y_true, n_bins: int = 15) -> float:
    """Expected calibration error over the predicted class confidence."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true).ravel()
    conf = p.max(1)
    pred = p.argmax(1)
    acc = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


# ---------------------------------------------------------------------- genus

# Genus -> family, hand-checked against the 33 species in FOR-species20K.
FAMILY = {
    "Abies": "Pinaceae", "Larix": "Pinaceae", "Picea": "Pinaceae",
    "Pinus": "Pinaceae", "Pseudotsuga": "Pinaceae",
    "Fagus": "Fagaceae", "Quercus": "Fagaceae",
    "Betula": "Betulaceae", "Carpinus": "Betulaceae", "Corylus": "Betulaceae",
    "Acer": "Sapindaceae", "Populus": "Salicaceae", "Fraxinus": "Oleaceae",
    "Crataegus": "Rosaceae", "Prunus": "Rosaceae", "Tilia": "Malvaceae",
    "Ulmus": "Ulmaceae", "Eucalyptus": "Myrtaceae", "Euonymus": "Celastraceae",
}


def family_of(species_names):
    """Family is not a field in the dataset; it is the genus mapped through FAMILY."""
    g = genus_of(species_names)
    return np.array([FAMILY.get(x, "?") for x in g])


def family_accuracy(true_species_names, pred_species_names):
    tf = family_of(true_species_names)
    pf = family_of(pred_species_names)
    return float((tf == pf).mean()) if len(tf) else float("nan")

def genus_of(species_names):
    """Genus is the first token of the binomial, which is what the dataset ships."""
    return np.array([str(s).split("_")[0] for s in species_names])


def genus_accuracy(species_names, y_true_idx, y_pred_idx, mask=None) -> float:
    g = genus_of(species_names)
    yt = g[np.asarray(y_true_idx)]
    yp = g[np.asarray(y_pred_idx)]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return float("nan")
        yt, yp = yt[m], yp[m]
    return float((yt == yp).mean())


def genus_topk(species_names, probs: np.ndarray, y_true_idx, k: int = 3) -> float:
    """Top-k accuracy at genus level: sum class masses per genus, then top-k."""
    g = genus_of(species_names)
    genera, ginv = np.unique(g, return_inverse=True)
    agg = np.zeros((probs.shape[0], len(genera)), dtype=np.float64)
    np.add.at(agg.T, ginv, probs.T)  # sum columns belonging to the same genus
    topk = np.argsort(-agg, axis=1)[:, :k]
    yt = ginv[np.asarray(y_true_idx)]
    return float((topk == yt[:, None]).any(1).mean())


# ------------------------------------------------------------------ self-test
def _self_test() -> int:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, got, want, tol=1e-9):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:<34} got {got:.6f} want {want:.6f}")

    print("[auroc]")
    check("perfect separation", auroc([1, 2, 3], [0, -1, -2]), 1.0)
    check("total inversion", auroc([0, 0, 0], [1, 1, 1]), 0.0)
    check("all ties", auroc([1, 1], [1, 1]), 0.5)
    check("hand computed", auroc(np.array([3, 4, 5]), np.array([1, 2])),
          (1 + 1 + 1) / 3 / 2 * 2)

    print("[fpr_at_tpr]")
    known = np.concatenate([rng.normal(2, 1, 1000), ])
    unknown = rng.normal(0, 1, 1000)
    f = fpr_at_tpr(known, unknown, 0.95)
    print(f"  info  fpr@95tpr = {f:.4f} (should be small for well separated)")

    print("[oscr]")
    ks = rng.normal(2, 1, 2000)
    kc = rng.random(2000) < 0.9
    us = rng.normal(0, 1, 1000)
    a = oscr_auc(ks, kc, us)
    print(f"  info  oscr_auc = {a:.4f}, oscr@5%fpr = {oscr_at_fpr(ks, kc, us):.4f}")
    ok &= 0.0 <= a <= 1.0
    # a perfect scorer with 100% closed-set accuracy must reach area ~1
    a2 = oscr_auc(np.r_[np.full(100, 10.0)], np.ones(100, bool), np.zeros(100))
    check("perfect scorer oscr", a2, 1.0, tol=1e-6)

    print("[osa]")
    thr = threshold_at_tpr(us, 0.95)
    v = osa(ks, kc, us, thr)
    print(f"  info  osa = {v:.4f} at thr={thr:.3f}")
    ok &= 0.0 <= v <= 1.0

    print("[ece]")
    p = np.zeros((1000, 3))
    y = rng.integers(0, 3, 1000)
    p[np.arange(1000), y] = 1.0                      # perfectly confident and correct
    check("perfect calibration", ece(p, y), 0.0, tol=1e-12)
    p2 = np.full((1000, 3), 1 / 3)
    print(f"  info  ece of uniform probs = {ece(p2, y):.4f} (acc 1/3, conf 1/3 -> ~0)")

    print("[macro / genus]")
    yt = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    check("macro recall", macro_recall(yt, yp), (0.5 + 1.0) / 2)
    names = ["Pinus_sylvestris", "Pinus_nigra", "Fagus_sylvatica"]
    check("genus acc", genus_accuracy(names, np.array([0, 1]), np.array([1, 2])), 0.5)
    pr = np.array([[0.4, 0.35, 0.25], [0.1, 0.2, 0.7]])
    # sample 0: Pinus mass .75 vs Fagus .25 -> right.  sample 1: true Pinus but .7 on
    # Fagus -> wrong.  So genus top-1 is 0.5, and top-2 must be 1.0.
    check("genus top1", genus_topk(names, pr, np.array([0, 1]), k=1), 0.5)
    check("genus top2", genus_topk(names, pr, np.array([0, 1]), k=2), 1.0)

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
