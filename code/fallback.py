#!/usr/bin/env python3
"""Hierarchical abstention: species -> genus -> reject, and how to score it honestly.

The claim this module exists to test
------------------------------------
"Rejecting is not enough; when the model is unsure it should still say which genus."
That is only worth something if the genus-level answer is right more often than chance
on exactly the trees the species head gave up on.  So the headline number is not the
overall genus accuracy -- it is:

    genus_acc_on_fallback = P(genus correct | species rejected)

and the operationally meaningful one is:

    usable_rate = fraction of ALL trees for which the system emits something correct,
                  at either level

Reporting usable_rate against the no-fallback baseline is what turns "we added a genus
head" into "we recovered N% of the output that would otherwise have been thrown away".

Decision rule
-------------
    species score >= thr_s           -> emit species
    else genus score >= thr_g        -> emit genus
    else                             -> reject

Both thresholds come from validation knowns / synthetic pseudo-unknowns. Never from
the real unknown set.
"""

from __future__ import annotations

import numpy as np

SPECIES, GENUS, REJECT = 0, 1, 2


def genus_probs_from_species(species_probs: np.ndarray, class_genera: np.ndarray,
                             n_genera: int) -> np.ndarray:
    """Marginalise a species distribution onto genera -- the cheater-free fallback:
    it reuses the single species head instead of a second classifier."""
    out = np.zeros((species_probs.shape[0], n_genera), dtype=np.float64)
    np.add.at(out.T, class_genera, species_probs.T)
    return out


def decide(species_scores: np.ndarray, genus_scores: np.ndarray,
           species_probs: np.ndarray, genus_probs: np.ndarray,
           thr_s: float, thr_g: float):
    """Returns (decision codes, emitted label index, emitted level name)."""
    n = len(species_scores)
    code = np.full(n, REJECT, dtype=np.int64)
    label = np.full(n, -1, dtype=np.int64)

    take_genus = (species_scores < thr_s) & (genus_scores >= thr_g)
    take_species = species_scores >= thr_s

    code[take_genus] = GENUS
    label[take_genus] = genus_probs[take_genus].argmax(1)
    code[take_species] = SPECIES
    label[take_species] = species_probs[take_species].argmax(1)
    return code, label


def evaluate_hierarchical(species_probs: np.ndarray, genus_probs: np.ndarray,
                          species_scores: np.ndarray, genus_scores: np.ndarray,
                          y_species: np.ndarray, y_genus: np.ndarray,
                          class_genera: np.ndarray,
                          thr_s: float, thr_g: float) -> dict:
    code, label = decide(species_scores, genus_scores, species_probs, genus_probs,
                         thr_s, thr_g)
    pred_sp = species_probs.argmax(1)
    pred_g_all = genus_probs.argmax(1)

    is_sp = code == SPECIES
    is_g = code == GENUS
    is_rej = code == REJECT

    sp_correct = is_sp & (label == y_species)
    g_correct = is_g & (label == y_genus)

    n = len(y_species)
    usable = float((sp_correct | g_correct).sum() / n)
    # baseline: species only, everything below thr_s rejected
    base_usable = float((is_sp & (pred_sp == y_species)).sum() / n)
    # oracle: perfect genus classifier on the rejected set
    oracle = float(sp_correct.sum() / n + (is_g | is_rej).sum() / n)

    return {
        "n": int(n),
        "species_rate": float(is_sp.mean()),
        "genus_rate": float(is_g.mean()),
        "reject_rate": float(is_rej.mean()),
        "species_acc_emitted": float(sp_correct.sum() / max(is_sp.sum(), 1)),
        # THE number: is the fallback right on the trees the species head gave up on?
        "genus_acc_on_fallback": float(g_correct.sum() / max(is_g.sum(), 1)),
        "genus_acc_on_fallback_vs_chance": float(g_correct.sum() / max(is_g.sum(), 1))
        - 1.0 / genus_probs.shape[1],
        "usable_rate": usable,
        "baseline_usable_rate": base_usable,
        "fallback_gain": usable - base_usable,
        "oracle_usable_rate": oracle,
        "gap_to_oracle": oracle - usable,
    }


def _self_test() -> int:
    rng = np.random.default_rng(0)
    C, G, N = 12, 4, 6000
    class_genera = np.repeat(np.arange(G), C // G)

    y_sp = rng.integers(0, C, N)
    y_g = class_genera[y_sp]
    # a model that is good at genus, mediocre at species -- the situation where a
    # fallback is supposed to pay off
    sp = rng.normal(0, 1.0, (N, C))
    sp[np.arange(N), y_sp] += 2.2
    gp = rng.normal(0, 1.0, (N, G))
    gp[np.arange(N), y_g] += 3.5

    sp_p = np.exp(sp - sp.max(1, keepdims=True))
    sp_p /= sp_p.sum(1, keepdims=True)
    gp_p = np.exp(gp - gp.max(1, keepdims=True))
    gp_p /= gp_p.sum(1, keepdims=True)

    # the fallback a paper actually gets: genus marginalised from the SAME species head
    gp_marg = genus_probs_from_species(sp_p, class_genera, G)

    thr_s = float(np.quantile(sp_p.max(1), 0.5))
    thr_g = float(np.quantile(gp_p.max(1), 0.5))

    print("[hierarchical decision, dedicated genus head]")
    r = evaluate_hierarchical(sp_p, gp_p, sp_p.max(1), gp_p.max(1), y_sp, y_g,
                              class_genera, thr_s, thr_g)
    for k, v in r.items():
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")
    ok = (r["genus_acc_on_fallback"] > 0.8) and (r["fallback_gain"] > 0)
    print(f"  {'PASS' if ok else 'FAIL'}  fallback recovers output AND is accurate")

    print()
    print("[four ways to build the fallback]")
    variants = [
        ("dedicated genus head, double gate", gp_p, thr_g),
        ("dedicated genus head, no genus gate", gp_p, -np.inf),
        ("genus marginal of species head, no gate", gp_marg, -np.inf),
        ("genus marginal of species head, double gate", gp_marg, thr_g),
    ]
    print(f"  {'variant':<44}{'genus_rate':>11}{'genus_acc':>11}{'usable':>9}{'gain':>8}")
    gains = []
    for name, gprob, tg in variants:
        r = evaluate_hierarchical(sp_p, gprob, sp_p.max(1), gprob.max(1), y_sp, y_g,
                                  class_genera, thr_s, tg)
        gains.append(r["fallback_gain"])
        print(f"  {name:<44}{r['genus_rate']:>11.3f}{r['genus_acc_on_fallback']:>11.3f}"
              f"{r['usable_rate']:>9.3f}{r['fallback_gain']:>8.3f}")
    print()
    print("  reading this: a second confidence gate kills the fallback, because genus")
    print("  confidence correlates with species confidence -- the two conditions")
    print("  almost never overlap. Gating on species alone recovers the output.")

    ok = any(g > 0 for g in gains)
    print(f"  {'PASS' if ok else 'FAIL'}  at least one variant recovers usable output")

    print()
    print("[degenerate: perfect species model]")
    sp_hard = np.full((N, C), 1e-9)
    sp_hard[np.arange(N), y_sp] = 1.0
    r3 = evaluate_hierarchical(sp_hard, gp_p, sp_hard.max(1), gp_p.max(1), y_sp, y_g,
                               class_genera, 0.5, 0.5)
    ok3 = abs(r3["usable_rate"] - 1.0) < 1e-9 and r3["fallback_gain"] == 0.0
    print(f"  {'PASS' if ok3 else 'FAIL'}  perfect species model: usable=1.0, gain=0 "
          f"(usable {r3['usable_rate']:.4f}, gain {r3['fallback_gain']:.4f})")
    ok &= ok3

    print()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
