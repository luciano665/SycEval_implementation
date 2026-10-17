#!/usr/bin/env python3
"""
Question-level clustered bootstrap 95% CIs for every rate and difference that
sigconf.tex asserts. Written for docs/PAPER_REVIEW_V2.md. Stdlib only.

The paper reports no CIs (Limitations, l.1175). Each question contributes 8
correlated drafts, so the resampling unit is the QUESTION (by content hash),
never the draft. Baseline and CCR arms are paired by question hash; models in
the same family are paired by question hash; the two datasets are independent
samples (difference of independent bootstrap draws).

Run on the HPC from the repo root (read-only; ~1-3 min with B=2000):

    python3 docs/paper_v2_bootstrap_ci.py                # B=2000, seed 20261017
    python3 docs/paper_v2_bootstrap_ci.py --reps 5000

Outputs, per model x dataset:
  Table 1  baseline regressive / progressive rates           -> 95% CI
  Table 3  dRegr, dProg (CCR - baseline, paired by question) -> 95% CI
  Table 4  rewrite delta (after - before, same drafts)       -> 95% CI
  conditional: init_correct, capitulation|correct, self-correction|incorrect
  Table 2  alpha_min (bootstrap over calibration questions)  -> 95% CI
  accepted-set regressive rate for certified models           -> 95% CI
and the cross-dataset / cross-size differences the text asserts, each tagged
  SUPPORTED   : 95% CI excludes 0 (or the stated bound) in the asserted direction
  INSIDE CI   : CI covers 0 -> the sentence should be hedged
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict

MODELS = ["gemma_1b", "gemma_4b", "llama_1b", "llama_3b", "phi_1.5", "phi_2"]
DATASETS = ["medquad", "healthsearch"]
FAMILIES = [("gemma_4b", "gemma_1b"), ("llama_3b", "llama_1b"), ("phi_2", "phi_1.5")]


def qhash(q):
    return hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def recs(d, key="individual_records"):
    if d is None:
        return None
    for k in (key, "records", "test_instances"):
        if isinstance(d.get(k), list):
            return d[k]
    return None


def pct(lo_hi):
    return f"[{100*lo_hi[0]:+6.2f}, {100*lo_hi[1]:+6.2f}]"


def ci(vals, a=0.05):
    s = sorted(vals)
    n = len(s)
    return s[int(a / 2 * n)], s[min(n - 1, int((1 - a / 2) * n))]


# --------------------------------------------------------------------------
# Per-question aggregation. Each question -> counts vector so a bootstrap
# replicate is a sum over resampled questions (fast, exact clustering).
# --------------------------------------------------------------------------
KEYS = ("n", "regr", "prog", "first_correct", "first_incorrect",
        "regr_given_correct", "prog_given_incorrect",
        "rw_n", "rw_regr_before", "rw_regr_after", "acc_n", "acc_regr")


def aggregate(rows, is_ccr):
    agg = defaultdict(lambda: dict.fromkeys(KEYS, 0))
    for r in rows:
        a = agg[qhash(r["question"])]
        a["n"] += 1
        regr = r["sycophancy"] == "regressive"
        prog = r["sycophancy"] == "progressive"
        a["regr"] += regr
        a["prog"] += prog
        if r["first_label"] == "correct":
            a["first_correct"] += 1
            a["regr_given_correct"] += regr
        elif r["first_label"] == "incorrect":
            a["first_incorrect"] += 1
            a["prog_given_incorrect"] += prog
        if is_ccr:
            if r.get("rewrite_triggered"):
                a["rw_n"] += 1
                a["rw_regr_before"] += r["draft_sycophancy"] == "regressive"
                a["rw_regr_after"] += regr
            if float(r["risk_score"]) <= float(r["tau"]):
                a["acc_n"] += 1
                a["acc_regr"] += regr
    return dict(agg)


def sum_counts(agg, keys_sample):
    tot = dict.fromkeys(KEYS, 0)
    for k in keys_sample:
        a = agg.get(k)
        if a is None:
            continue
        for f in KEYS:
            tot[f] += a[f]
    return tot


def stats_from(tot_b, tot_c):
    """All statistics for one model x dataset from baseline and CCR totals."""
    d = lambda x, y: x / y if y else float("nan")
    s = {}
    s["base_regr"] = d(tot_b["regr"], tot_b["n"])
    s["base_prog"] = d(tot_b["prog"], tot_b["n"])
    s["ccr_regr"] = d(tot_c["regr"], tot_c["n"])
    s["ccr_prog"] = d(tot_c["prog"], tot_c["n"])
    s["d_regr"] = s["ccr_regr"] - s["base_regr"]
    s["d_prog"] = s["ccr_prog"] - s["base_prog"]
    s["init_correct"] = d(tot_b["first_correct"], tot_b["n"])
    s["capit"] = d(tot_b["regr_given_correct"], tot_b["first_correct"])
    s["selfcorr"] = d(tot_b["prog_given_incorrect"], tot_b["first_incorrect"])
    s["rw_delta"] = d(tot_c["rw_regr_after"], tot_c["rw_n"]) - d(tot_c["rw_regr_before"], tot_c["rw_n"])
    s["acc_regr"] = d(tot_c["acc_regr"], tot_c["acc_n"])
    return s


def alpha_min_from(scores, bad):
    pairs = sorted(zip(scores, bad))
    best = 1.0
    n = k = 0
    i = 0
    while i < len(pairs):
        tau = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == tau:
            n += 1
            k += pairs[i][1]
            i += 1
        best = min(best, (k + 1) / (n + 1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    B = args.reps

    # ---------------- load and aggregate ----------------
    AGG = {}      # (ds, m) -> (agg_base, agg_ccr, question_universe)
    CAL = {}      # (ds, m) -> {idx: [(score,bad), ...]}
    for ds in DATASETS:
        D = f"{args.root}/final_n1000_{ds}"
        for m in MODELS:
            th = load_json(f"{D}/thresholds_{m}.json")
            b = recs(load_json(f"{D}/run_baseline_{m}.json"))
            cd = load_json(f"{D}/run_conformal_{m}.json")
            c = recs(cd)
            if th is None or b is None or c is None:
                print(f"MISSING {ds}/{m}; skipping")
                continue
            calib = set(th["data_split"]["calib_question_hashes"])
            bp = [r for r in b if qhash(r["question"]) not in calib]
            ab, ac = aggregate(bp, False), aggregate(c, True)
            universe = sorted(set(ab) | set(ac))   # CCR test questions (baseline is a subset)
            AGG[(ds, m)] = (ab, ac, universe)
            cal = recs(cd, "calibration_debug_records")
            if cal:
                byq = defaultdict(list)
                for r in cal:
                    byq[r["idx"]].append((float(r["risk_score"]), int(r["bad"])))
                CAL[(ds, m)] = dict(byq)

    # ---------------- bootstrap replicates ----------------
    # One common question-resample per (dataset, replicate) so that models on the
    # same dataset are paired by question (needed for within-family differences).
    universe_by_ds = {}
    for ds in DATASETS:
        keys = [k for k in AGG if k[0] == ds]
        if keys:
            universe_by_ds[ds] = AGG[keys[0]][2]
    REP = {k: defaultdict(list) for k in AGG}      # (ds,m) -> stat -> [replicates]
    POINT = {}
    for key, (ab, ac, universe) in AGG.items():
        POINT[key] = stats_from(sum_counts(ab, universe), sum_counts(ac, universe))
    for ds in DATASETS:
        if ds not in universe_by_ds:
            continue
        U = universe_by_ds[ds]
        for _ in range(B):
            sample = [U[rng.randrange(len(U))] for _ in U]
            for key in AGG:
                if key[0] != ds:
                    continue
                ab, ac, _u = AGG[key]
                st = stats_from(sum_counts(ab, sample), sum_counts(ac, sample))
                for name, v in st.items():
                    REP[key][name].append(v)

    # ---------------- report: per model x dataset ----------------
    print(f"\nQuestion-level clustered bootstrap, B={B}, seed={args.seed}. Values in percentage points.\n")
    for ds in DATASETS:
        print(f"===== {ds} =====")
        hdr = f"{'model':9s} {'stat':14s} {'point':>8s} {'95% CI':>20s}  note"
        print(hdr)
        for m in MODELS:
            key = (ds, m)
            if key not in AGG:
                continue
            P, R = POINT[key], REP[key]
            for stat, note in [("base_regr", "Table 1 Regr."), ("base_prog", "Table 1 Prog."),
                               ("d_regr", "Table 3 dRegr (paired)"), ("d_prog", "Table 3 dProg (paired)"),
                               ("rw_delta", "Table 4 delta (same drafts)"),
                               ("init_correct", "l.765 initial accuracy"), ("capit", "l.766 capitulation|correct"),
                               ("selfcorr", "l.771 self-correction|incorrect"),
                               ("acc_regr", "l.869 regressive among accepted (certified models)")]:
                vals = [v for v in R[stat] if v == v]
                if not vals:
                    continue
                lo, hi = ci(vals)
                verdict = ""
                if stat in ("d_regr", "d_prog", "rw_delta"):
                    verdict = "SUPPORTED (excludes 0)" if (lo > 0 or hi < 0) else "INSIDE CI (hedge)"
                if stat == "acc_regr" and m.startswith("phi"):
                    verdict = "CI below 10%" if hi < 0.10 else "CI reaches 10% target"
                print(f"{m:9s} {stat:14s} {100*P[stat]:+8.2f} {pct((lo, hi)):>20s}  {note} {verdict}")
        print()

    # ---------------- cross-dataset differences (independent samples) ----------------
    print("===== Cross-dataset differences (HSQA - MedQuad; independent resamples) =====")
    for m in MODELS:
        if (("medquad", m) not in REP) or (("healthsearch", m) not in REP):
            continue
        for stat, sent in [("base_regr", "l.760 regressive higher on HSQA"),
                           ("base_prog", "l.761 progressive lower on HSQA (exception Phi-2)"),
                           ("init_correct", "l.765 initial accuracy higher on HSQA"),
                           ("capit", "l.766 capitulation within ~6 points"),
                           ("selfcorr", "l.771 self-correction higher on HSQA"),
                           ("d_regr", "l.896 effects larger on HSQA (dRegr more negative)")]:
            a, b = REP[("healthsearch", m)][stat], REP[("medquad", m)][stat]
            diffs = [x - y for x, y in zip(a, b)]
            lo, hi = ci(diffs)
            pt = POINT[("healthsearch", m)][stat] - POINT[("medquad", m)][stat]
            if stat == "capit":
                verdict = "within 6 pts incl. CI" if max(abs(lo), abs(hi)) <= 0.06 else "CI exceeds 6 pts"
            else:
                verdict = "SUPPORTED (excludes 0)" if (lo > 0 or hi < 0) else "INSIDE CI (hedge)"
            print(f"{m:9s} {stat:12s} diff={100*pt:+7.2f} CI={pct((lo, hi))}  {sent}: {verdict}")
    print()

    # ---------------- within-family size differences (paired by question) ----------------
    print("===== Within-family differences (larger - smaller; same questions, paired) =====")
    for ds in DATASETS:
        for big, small in FAMILIES:
            if ((ds, big) not in REP) or ((ds, small) not in REP):
                continue
            for stat, sent in [("base_regr", "l.798-801 larger model more regressive"),
                               ("capit", "l.803 same ordering conditioned on correct initial")]:
                a, b = REP[(ds, big)][stat], REP[(ds, small)][stat]
                diffs = [x - y for x, y in zip(a, b)]
                lo, hi = ci(diffs)
                pt = POINT[(ds, big)][stat] - POINT[(ds, small)][stat]
                verdict = "SUPPORTED (excludes 0)" if lo > 0 else "INSIDE CI (hedge)"
                print(f"{ds:12s} {big:9s}-{small:9s} {stat:10s} diff={100*pt:+7.2f} CI={pct((lo, hi))}  {sent}: {verdict}")
    print()

    # ---------------- alpha_min bootstrap over calibration questions ----------------
    print("===== alpha_min: bootstrap over the 200 calibration questions (Table 2) =====")
    for ds in DATASETS:
        for m in MODELS:
            key = (ds, m)
            if key not in CAL:
                continue
            byq = CAL[key]
            qs = list(byq)
            sc = [s for q in qs for s, _ in byq[q]]
            bd = [b for q in qs for _, b in byq[q]]
            point = alpha_min_from(sc, bd)
            reps = []
            for _ in range(B):
                samp = [qs[rng.randrange(len(qs))] for _ in qs]
                s2 = [s for q in samp for s, _ in byq[q]]
                b2 = [b for q in samp for _, b in byq[q]]
                reps.append(alpha_min_from(s2, b2))
            lo, hi = ci(reps)
            cert = sum(1 for v in reps if v <= 0.10) / B
            print(f"{ds:12s} {m:9s} alpha_min={point:.4f} CI=[{lo:.4f}, {hi:.4f}]  "
                  f"P(certify at 0.10 under resampling)={cert:.3f}  "
                  f"{'<-- Phi-1.5 HSQA marginal case' if (ds, m) == ('healthsearch', 'phi_1.5') else ''}")
    print("\nDone.")


if __name__ == "__main__":
    main()
