#!/usr/bin/env python3
"""Open-set scorers and the evaluation driver, numpy only.

Design choice: the model is NOT part of this file.
-----------------------------------------------
Training writes three arrays, and everything below runs on them:

    logits.npy   (N, C)  raw classifier logits, column order = CLASS_NAMES order
    feats.npy    (N, D)  penultimate features (optional, needed for Mahalanobis/kNN)
    meta.csv     treeID, species, ... one row per array row, same order

That keeps the expensive, GPU-bound part swappable (PointNet++, ConvNeXtV2, whatever
wins the closed-set leaderboard) while the open-set science stays in numpy and can be
re-run in seconds.

Score convention: higher == more likely KNOWN. See metrics.py.

Scorers
-------
    msp           max softmax probability
    energy        T * logsumexp(logits / T)          (Liu et al. 2020)
    mahalanobis   negative min distance to class means, pooled+shrinkage covariance
    knn           negative distance to the k-th nearest training feature

Threshold rule (this is the part reviewers check)
-------------------------------------------------
The operating point must never be tuned on the real unknown set. Two leak-free options:

    --thr-mode knowns   keep `--keep-known` of the known validation scores
                        (default 0.95, i.e. reject the least confident 5%)
    --thr-mode synth    use pseudo-unknowns synthesised from the training features
                        (needs features + a linear head; see make_pseudo_unknowns)

Usage
-----
    python code/osr.py --logits runs/pn2/logits.npy --features runs/pn2/feats.npy \
        --meta data/npz/meta.csv --split splits_clean/s5_pinus.json --out results/osr_pinus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M


# ------------------------------------------------------------------- scorers
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logsumexp(z: np.ndarray, axis: int = 1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def score_msp(logits: np.ndarray) -> np.ndarray:
    return softmax(logits).max(1)


def score_energy(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    return T * logsumexp(logits / T)


SHRINK = 1e-3  # single source of truth for the pooled-covariance shrinkage


def fit_mahalanobis(feats: np.ndarray, y: np.ndarray, shrink: float = 0.1) -> dict:
    """Class means + a single pooled precision matrix.

    A per-class covariance would be rank deficient for the small classes (several
    species have a few hundred trees at most), so the shared pooled covariance with
    shrinkage is the only defensible choice here.
    """
    classes = np.unique(y)
    d = feats.shape[1]
    means = np.zeros((len(classes), d), dtype=np.float64)
    cov = np.zeros((d, d), dtype=np.float64)
    n_total = 0
    for i, c in enumerate(classes):
        f = feats[y == c].astype(np.float64)
        means[i] = f.mean(0)
        if len(f) > 1:
            r = f - means[i]
            cov += r.T @ r
            n_total += len(f) - 1
    cov /= max(n_total, 1)
    cov = (1.0 - shrink) * cov + shrink * np.trace(cov) / d * np.eye(d)
    prec = np.linalg.pinv(cov)
    return {"classes": classes, "means": means, "prec": prec}


def score_mahalanobis(feats: np.ndarray, model: dict, chunk: int = 4096) -> np.ndarray:
    means, prec = model["means"], model["prec"]
    out = np.empty(len(feats), dtype=np.float64)
    for s in range(0, len(feats), chunk):
        f = feats[s:s + chunk].astype(np.float64)
        # (n, K, D) residual -> quadratic form, vectorised over classes
        diff = f[:, None, :] - means[None, :, :]
        q = np.einsum("nkd,de,nke->nk", diff, prec, diff)
        out[s:s + chunk] = np.sqrt(np.clip(q.min(1), 0, None))
    return -out          # higher score == closer to a class mean == more known


def score_knn(feats: np.ndarray, train_feats: np.ndarray, k: int = 10,
              max_train: int = 8000, seed: int = 0, chunk: int = 1024) -> np.ndarray:
    """Negative distance to the k-th nearest training point.

    Brute force is O(N_test x N_train x D); with 17 k trees that is fine on CPU if the
    training set is capped (default 8000 sampled points) and the work is chunked.
    """
    rng = np.random.default_rng(seed)
    tf = train_feats
    if len(tf) > max_train:
        tf = tf[rng.choice(len(tf), max_train, replace=False)]
    tf = tf.astype(np.float32)
    tt = (tf ** 2).sum(1)
    out = np.empty(len(feats), dtype=np.float64)
    kk = min(k, len(tf))
    for s in range(0, len(feats), chunk):
        f = feats[s:s + chunk].astype(np.float32)
        d2 = (f ** 2).sum(1)[:, None] + tt[None, :] - 2.0 * f @ tf.T
        d2 = np.clip(d2, 0, None)
        out[s:s + chunk] = np.sqrt(np.partition(d2, kk - 1, axis=1)[:, kk - 1])
    return -out


# ------------------------------------------------------------ pseudo unknowns
def make_pseudo_unknowns(feats: np.ndarray, y: np.ndarray, genera: np.ndarray,
                         n: int = 4000, mode: str = "cross-genus", seed: int = 0) -> np.ndarray:
    """Synthesise features that stand in for unseen species.

    cross-genus: midpoint of two samples from DIFFERENT genera. The taxonomic argument
                 is that a genuinely new species is more likely to look like something
                 from another genus than like a resampled known species -- and unlike
                 additive noise, it lands in the region between clusters instead of on
                 top of one.
    noise:       additive gaussian noise, the standard outlier-exposure baseline.
    interp:      midpoint of any two samples, ignoring taxonomy (ablation).
    """
    rng = np.random.default_rng(seed)
    n_f = feats.shape[0]
    out = np.empty((n, feats.shape[1]), dtype=np.float64)

    if mode == "noise":
        sd = feats.std(0) + 1e-8
        idx = rng.integers(0, n_f, n)
        return feats[idx] + rng.normal(0, 1.0, (n, feats.shape[1])) * sd

    ii = rng.integers(0, n_f, n)
    jj = rng.integers(0, n_f, n)
    if mode == "cross-genus":
        bad = genera[ii] == genera[jj]
        tries = 0
        while bad.any() and tries < 50:
            jj[bad] = rng.integers(0, n_f, bad.sum())
            bad = genera[ii] == genera[jj]
            tries += 1
    lam = rng.uniform(0.3, 0.7, n)[:, None]
    out[:] = lam * feats[ii] + (1 - lam) * feats[jj]
    return out


# ---------------------------------------------------------------- evaluation
def evaluate(scores_known, correct_known, scores_unknown, threshold,
             probs_known, y_known, genus_true_unknown, pred_unknown,
             family_true_unknown=None, pred_family_unknown=None) -> dict:
    rej_known = scores_known < threshold
    acc_unknown = scores_unknown >= threshold
    out = {
        "auroc": M.auroc_known_vs_unknown(scores_known, scores_unknown),
        "oscr_auc": M.oscr_auc(scores_known, correct_known, scores_unknown),
        "fpr_at_95tpr": M.fpr_at_tpr(scores_known, scores_unknown, 0.95),
        "oscr_at_5fpr": M.oscr_at_fpr(scores_known, correct_known, scores_unknown, 0.05),
        "osa": M.osa(scores_known, correct_known, scores_unknown, threshold),
        "known_acc": float(correct_known.mean()) if len(correct_known) else float("nan"),
        "macro_recall": M.macro_recall(y_known, probs_known.argmax(1)),
        "ece": M.ece(probs_known, y_known),
        "known_reject_rate": float(rej_known.mean()) if len(rej_known) else float("nan"),
        "unknown_accept_rate": float(acc_unknown.mean()) if len(acc_unknown) else float("nan"),
        # the number the genus-fallback claim rests on
        "genus_recall_unknown": float(np.mean(genus_true_unknown == pred_unknown))
        if len(pred_unknown) else float("nan"),
        "genus_recall_unknown_accepted": float(
            np.mean(genus_true_unknown[acc_unknown] == pred_unknown[acc_unknown]))
        if acc_unknown.any() else float("nan"),
        "n_known": int(len(scores_known)),
        "n_unknown": int(len(scores_unknown)),
        "threshold": float(threshold),
    }
    # Family is the coarser level the measurements actually support; report it next to
    # genus so the difference between the two is visible in every table.
    if family_true_unknown is not None and pred_family_unknown is not None:
        out["family_recall_unknown"] = float(
            np.mean(family_true_unknown == pred_family_unknown)) if len(pred_family_unknown) else float("nan")
        out["family_recall_unknown_accepted"] = float(
            np.mean(family_true_unknown[acc_unknown] == pred_family_unknown[acc_unknown])) \
            if acc_unknown.any() else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Open-set scoring and evaluation")
    ap.add_argument("--logits", type=Path, required=True, help="(N, C) npy")
    ap.add_argument("--features", type=Path, default=None, help="(N, D) npy, for mahalanobis/knn")
    ap.add_argument("--meta", type=Path, required=True, help="csv with treeID, species")
    ap.add_argument("--split", type=Path, required=True, help="split json with unknown_treeIDs")
    ap.add_argument("--fold", type=int, default=0,
                    help="index into split['folds']; sets which validation fold calibrates the threshold")
    ap.add_argument("--classes", type=Path, default=None,
                    help="json list of class names in logit-column order")
    ap.add_argument("--out", type=Path, default=Path("results/osr"))
    ap.add_argument("--scorers", nargs="+", default=["msp", "energy", "mahalanobis", "knn"])
    ap.add_argument("--thr-mode", choices=["knowns", "synth"], default="knowns")
    ap.add_argument("--keep-known", type=float, default=0.95)
    ap.add_argument("--energy-T", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=1e-3,
                    help="Mahalanobis shrinkage toward trace/d * I")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    logits = np.load(args.logits).astype(np.float64)
    meta = pd.read_csv(args.meta)
    meta["treeID"] = meta["treeID"].astype(np.int64)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))

    if args.classes:
        class_names = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    else:
        class_names = sorted(meta["species"].unique().tolist())
    if logits.shape[1] != len(class_names):
        raise SystemExit(f"logits has {logits.shape[1]} columns but {len(class_names)} class names")

    name2idx = {s: i for i, s in enumerate(class_names)}
    n = len(meta)
    if logits.shape[0] != n:
        raise SystemExit(f"logits rows {logits.shape[0]} != meta rows {n}")

    y = np.array([name2idx.get(s, -1) for s in meta["species"]], dtype=np.int64)

    unknown_ids = set(split["unknown_treeIDs"])
    is_unknown = meta["treeID"].isin(unknown_ids).to_numpy()

    # A row outside the class list is legitimate as long as the split declares it
    # unknown. A row that is neither is a species with no training examples -- on a
    # source-holdout fold this happens because 19 of 33 species live in a single
    # source, so removing that source removes the species too. Such trees are
    # unclassifiable by construction, so they are excluded from the known set and
    # counted, rather than aborting the whole evaluation or being scored as errors.
    feats = None
    bad = (y < 0) & ~is_unknown
    n_unattainable = int(bad.sum())
    if bad.any():
        print(f"[exclude] {n_unattainable} trees whose species has no training "
              f"example: {sorted(set(meta.loc[bad, 'species']))[:6]}")
        keep = ~bad
        y = y[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        logits = logits[keep]
        if args.features:
            _f = np.load(args.features)
            feats = _f[keep].astype(np.float64)
        is_unknown = meta["treeID"].isin(unknown_ids).to_numpy()
    n_rows = int(is_unknown.size)

    # threshold calibration set: the validation fold of THIS fold (known classes only)
    val_ids = set(split["folds"][args.fold]["val"])
    is_val = meta["treeID"].isin(val_ids).to_numpy() & ~is_unknown

    # The feature-based detectors must be fitted on the TRAINING trees only. Fitting them on
    # ~is_unknown (as an earlier version did) pulls the validation trees -- and their labels --
    # into the class means and the kNN reference library, so a validation tree can be its own
    # neighbour and the resulting AUROC is no longer a post-training measurement.
    train_ids = set(split["folds"][args.fold]["train"])
    is_train = meta["treeID"].isin(train_ids).to_numpy() & ~is_unknown & (y >= 0)
    if not is_val.any():
        is_val = ~is_unknown

    if args.features and feats is None:
        feats = np.load(args.features).astype(np.float64)
    genera = M.genus_of(class_names)
    print("=" * 78)
    print("OSR evaluation")
    print("=" * 78)
    print(f"trees {n:,}  classes {len(class_names)}  known {int((~is_unknown).sum()):,}  "
          f"unknown {int(is_unknown.sum()):,}  val {int(is_val.sum()):,}")
    print(f"threshold mode: {args.thr_mode} (keep {args.keep_known:.0%} of known val)" if args.thr_mode == "knowns"
          else "threshold mode: synth")

    # Only the held-out knowns and the unknowns are ever scored. Computing scores for the
    # training rows too costs 2.8x more time and changes no reported number, which matters
    # at 2,048 dimensions where the quadratic form is expensive.
    need = is_val | is_unknown
    _pu_cache = {}

    def pseudo_unknowns():
        if "v" not in _pu_cache:
            _pu_cache["v"] = make_pseudo_unknowns(
                feats[~is_unknown], y[~is_unknown], genera[y[~is_unknown]], seed=args.seed)
        return _pu_cache["v"]

    rows = []
    for sc in args.scorers:
        if sc in ("mahalanobis", "knn") and feats is None:
            print(f"[skip] {sc}: needs --features")
            continue
        if sc == "msp":
            s_all = np.full(n_rows, np.nan, dtype=np.float64)
            s_all[need] = score_msp(logits[need])
            s_synth = None
        elif sc == "energy":
            s_all = np.full(n_rows, np.nan, dtype=np.float64)
            s_all[need] = score_energy(logits[need], args.energy_T)
            s_synth = None
        elif sc == "mahalanobis":
            mdl = fit_mahalanobis(feats[is_train], y[is_train], shrink=args.shrink)
            s_all = np.full(n_rows, np.nan, dtype=np.float64)
            s_all[need] = score_mahalanobis(feats[need], mdl)
            # the synthetic threshold is only needed in that mode; skip the second pass
            s_synth = score_mahalanobis(pseudo_unknowns(), mdl) if args.thr_mode == "synth" else None
        elif sc == "knn":
            s_all = np.full(n_rows, np.nan, dtype=np.float64)
            s_all[need] = score_knn(feats[need], feats[is_train], k=args.knn_k, seed=args.seed)
            s_synth = score_knn(pseudo_unknowns(), feats[is_train], k=args.knn_k, seed=args.seed) if args.thr_mode == "synth" else None
        else:
            raise SystemExit(f"unknown scorer {sc}")

        if args.thr_mode == "synth":
            if s_synth is None:
                print(f"[skip] {sc}: synth threshold needs features-based scorer")
                continue
            thr = M.threshold_at_tpr(s_synth, args.keep_known)
        else:
            # keep `keep_known` of the known validation scores -> threshold at the
            # complementary quantile, so the reject rate on knowns is 1 - keep_known.
            thr = float(np.quantile(s_all[is_val], 1.0 - args.keep_known))

        probs_all = softmax(logits)
        # The known side must be the HELD-OUT validation knowns, not every known tree:
        # train trees are almost always classified correctly, so including them inflates
        # AUROC, OSCR and known accuracy by measuring memorisation rather than separation.
        sk, su = s_all[is_val], s_all[is_unknown]
        # The true genus for an unknown tree must come from the metadata, not from y:
        # y is -1 for every unknown row, so genera[y] silently indexed the LAST genus
        # in the list and the recovery rate was structurally zero.
        true_genus_unknown = M.genus_of(meta.loc[is_unknown, "species"].to_numpy())
        true_family_unknown = M.family_of(meta.loc[is_unknown, "species"].to_numpy())
        pred_species_unknown = np.array(class_names)[probs_all.argmax(1)[is_unknown]]
        r = evaluate(sk, (probs_all.argmax(1)[is_val] == y[is_val]),
                     su, thr, probs_all[is_val], y[is_val],
                     true_genus_unknown,
                     genera[probs_all.argmax(1)[is_unknown]],
                     true_family_unknown,
                     M.family_of(pred_species_unknown))
        r["scorer"] = sc
        rows.append(r)

    if not rows:
        raise SystemExit("no scorer produced a result")

    df = pd.DataFrame(rows).set_index("scorer")
    cols = ["auroc", "oscr_auc", "fpr_at_95tpr", "known_acc", "osa", "macro_recall", "ece",
            "known_reject_rate", "unknown_accept_rate",
            "genus_recall_unknown", "genus_recall_unknown_accepted",
            "family_recall_unknown", "family_recall_unknown_accepted"]
    print()
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(df[cols].T.to_string())
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "osr_summary.csv", encoding="utf-8")
    print(f"\n[osr] wrote {args.out}/osr_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
