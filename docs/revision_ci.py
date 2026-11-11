#!/usr/bin/env python3
"""
Question-level clustered bootstrap 95% CIs for the numbers docs/REVISION_PLAN.md
proposes to add or hedge in sigconf.tex. Stdlib only, read-only, no LLM calls.

What it prints, per model x dataset (final N=1000 suite):
  A. Table 3: dRegr, dProg, dErr (CCR - baseline, all test instances), each
     with a 95% CI, paired by question across the two arms.
  B. Outcome split among initially correct answers, for both arms:
     share ending incorrect / erroneous / correct, plus (CCR only) the share
     whose final answer is literally "(No valid claims found)" (withheld) and
     how many of those the judge labelled erroneous.
     Also the CCR - baseline difference in each share with its CI. This is
     the source for "26.3% vs 4.9%" (Gemma-3 4B, HealthSearchQA).
  C. alpha_min diagnostics from the calibration records: alpha_min, the
     accepted count n(tau*) and regressive count k(tau*) where it is
     attained, the accept-everything value (k_all+1)/(n_all+1), a bootstrap
     CI over the 200 calibration questions, and the share of resamples in
     which the model certifies at alpha = 0.10.

Each question contributes up to 8 correlated drafts, so the resampling unit
is the QUESTION (content hash), never the draft. Baseline and CCR arms are
paired by question hash; the baseline arm is restricted to test-split
questions exactly as analyze_final_n1000_headline.py does.

Record fields used (same ones docs/paper_v2_recompute.py ran on):
  baseline individual_records: question, first_label, after_label, sycophancy
  CCR      individual_records: the same, plus final_answer
  CCR      calibration_debug_records: idx, risk_score, bad
  thresholds_<m>.json: data_split.calib_question_hashes

Run on the HPC from the repo root (login node is fine; about 1-2 minutes):

    python3 docs/revision_ci.py | tee revision_ci_output.txt
    python3 docs/revision_ci.py --reps 5000
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
EMPTY = "(No valid claims found)"  # run_conformal_v2.py:99

# Per-question count vector (same layout for both arms).
F = ("n", "regr", "prog", "err", "c0", "c0_inc", "c0_err", "c0_cor", "c0_withheld",
     "c0_withheld_err")
IX = {name: i for i, name in enumerate(F)}
ZERO = (0,) * len(F)


def qhash(q):
    """Same content id as run_conformal_v2._question_hash."""
    return hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def records(d):
    if d is None:
        return None
    for k in ("individual_records", "records", "test_instances"):
        if isinstance(d.get(k), list):
            return d[k]
    return None


def aggregate(rows):
    """question hash -> count vector."""
    agg = defaultdict(lambda: [0] * len(F))
    for r in rows:
        a = agg[qhash(r["question"])]
        a[IX["n"]] += 1
        a[IX["regr"]] += r["sycophancy"] == "regressive"
        a[IX["prog"]] += r["sycophancy"] == "progressive"
        a[IX["err"]] += r["after_label"] == "erroneous"
        if r["first_label"] == "correct":
            a[IX["c0"]] += 1
            a[IX["c0_inc"]] += r["after_label"] == "incorrect"
            a[IX["c0_err"]] += r["after_label"] == "erroneous"
            a[IX["c0_cor"]] += r["after_label"] == "correct"
            fa = r.get("final_answer")
            withheld = isinstance(fa, str) and fa.strip() == EMPTY
            a[IX["c0_withheld"]] += withheld
            a[IX["c0_withheld_err"]] += withheld and r["after_label"] == "erroneous"
    return {k: tuple(v) for k, v in agg.items()}


def total(agg, sample):
    rows = [agg.get(q, ZERO) for q in sample]
    return [sum(col) for col in zip(*rows)] if rows else list(ZERO)


def div(x, y):
    return x / y if y else float("nan")


def stats(tb, tc):
    g = lambda t, name: t[IX[name]]
    s = {}
    for name in ("regr", "prog", "err"):
        s["base_" + name] = div(g(tb, name), g(tb, "n"))
        s["ccr_" + name] = div(g(tc, name), g(tc, "n"))
        s["d_" + name] = s["ccr_" + name] - s["base_" + name]
    for name in ("c0_inc", "c0_err", "c0_cor"):
        s["base_" + name] = div(g(tb, name), g(tb, "c0"))
        s["ccr_" + name] = div(g(tc, name), g(tc, "c0"))
        s["d_" + name] = s["ccr_" + name] - s["base_" + name]
    s["ccr_c0_withheld"] = div(g(tc, "c0_withheld"), g(tc, "c0"))
    # Of the withheld answers, the share the judge labelled erroneous.
    s["ccr_withheld_err"] = div(g(tc, "c0_withheld_err"), g(tc, "c0_withheld"))
    return s


def ci(vals, a=0.05):
    v = sorted(x for x in vals if x == x)
    if not v:
        return float("nan"), float("nan")
    n = len(v)
    return v[int(a / 2 * n)], v[min(n - 1, int((1 - a / 2) * n))]


def fmt(point, lo, hi, signed=False):
    p = f"{100 * point:+.1f}" if signed else f"{100 * point:.1f}"
    return f"{p} [{100 * lo:+.1f}, {100 * hi:+.1f}]" if signed else f"{p} [{100 * lo:.1f}, {100 * hi:.1f}]"


def alpha_scan(pairs):
    """pairs: (risk_score, bad). Returns (alpha_min, n*, k*, accept_all)."""
    pairs = sorted(pairs)
    best, bn, bk = 1.0, 0, 0
    n = k = 0
    i = 0
    while i < len(pairs):
        tau = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == tau:
            n += 1
            k += pairs[i][1]
            i += 1
        v = (k + 1) / (n + 1)
        if v < best:
            best, bn, bk = v, n, k
    return best, bn, bk, ((k + 1) / (n + 1) if n else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    B = args.reps

    AGG, CAL, NOTE = {}, {}, {}
    for ds in DATASETS:
        d = f"{args.root}/final_n1000_{ds}"
        for m in MODELS:
            th = load(f"{d}/thresholds_{m}.json")
            base = records(load(f"{d}/run_baseline_{m}.json"))
            cd = load(f"{d}/run_conformal_{m}.json")
            ccr = records(cd)
            if th is None or base is None or ccr is None:
                print(f"MISSING {ds}/{m}; skipped")
                continue
            calib = set(th["data_split"]["calib_question_hashes"])
            base_test = [r for r in base if qhash(r["question"]) not in calib]
            ab, ac = aggregate(base_test), aggregate(ccr)
            AGG[(ds, m)] = (ab, ac)
            NOTE[(ds, m)] = (len(base_test), len(ccr), len(ab), len(ac))
            cal = cd.get("calibration_debug_records")
            if isinstance(cal, list) and cal:
                byq = defaultdict(list)
                for r in cal:
                    byq[r["idx"]].append((float(r["risk_score"]), int(r["bad"])))
                CAL[(ds, m)] = dict(byq)

    if not AGG:
        raise SystemExit("No result files found under " + args.root)

    # One common question resample per (dataset, replicate), so both arms of a
    # model see the same questions (paired) in every replicate.
    POINT, REP = {}, {k: defaultdict(list) for k in AGG}
    for ds in DATASETS:
        keys = [k for k in AGG if k[0] == ds]
        if not keys:
            continue
        universe = sorted(set().union(*[set(AGG[k][1]) for k in keys]))
        for k in keys:
            ab, ac = AGG[k]
            POINT[k] = stats(total(ab, universe), total(ac, universe))
        for _ in range(B):
            sample = [universe[rng.randrange(len(universe))] for _ in universe]
            for k in keys:
                ab, ac = AGG[k]
                for name, v in stats(total(ab, sample), total(ac, sample)).items():
                    REP[k][name].append(v)

    print(f"Question-level clustered bootstrap, B={B}, seed={args.seed}. "
          f"Percentages; brackets are 95% CIs.")
    print("Sanity: instance and question counts (baseline test rows, CCR rows, "
          "baseline questions, CCR questions):")
    for k in sorted(NOTE):
        print(f"  {k[0]:12s} {k[1]:9s} {NOTE[k]}")

    print("\n===== A. Table 3 deltas (CCR - baseline, all test instances) =====")
    print("'*' marks a CI that excludes 0.")
    for ds in DATASETS:
        print(f"-- {ds}")
        for m in MODELS:
            k = (ds, m)
            if k not in POINT:
                continue
            cells = []
            for name in ("d_regr", "d_prog", "d_err"):
                lo, hi = ci(REP[k][name])
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(f"{name}={fmt(POINT[k][name], lo, hi, signed=True)}{star}")
            print(f"  {m:9s} " + "  ".join(cells))

    print("\n===== B. Outcome of initially correct answers (share of that arm's "
          "initially correct instances) =====")
    for ds in DATASETS:
        print(f"-- {ds}")
        for m in MODELS:
            k = (ds, m)
            if k not in POINT:
                continue
            P, R = POINT[k], REP[k]
            for arm in ("base", "ccr"):
                cells = [f"{lab}={fmt(P[f'{arm}_c0_{lab}'], *ci(R[f'{arm}_c0_{lab}']))}"
                         for lab in ("inc", "err", "cor")]
                if arm == "ccr":
                    cells.append(f"withheld={fmt(P['ccr_c0_withheld'], *ci(R['ccr_c0_withheld']))}")
                    cells.append(f"(of withheld, judged erroneous: {100 * P['ccr_withheld_err']:.1f})")
                print(f"  {m:9s} {arm:4s} " + "  ".join(cells))
            cells = []
            for lab in ("inc", "err", "cor"):
                lo, hi = ci(R[f"d_c0_{lab}"])
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(f"d_{lab}={fmt(P[f'd_c0_{lab}'], lo, hi, signed=True)}{star}")
            print(f"  {m:9s} diff " + "  ".join(cells))

    print("\n===== C. alpha_min diagnostics (calibration records; bootstrap over "
          "calibration questions) =====")
    if not CAL:
        print("  no calibration_debug_records found; section skipped")
    for ds in DATASETS:
        for m in MODELS:
            k = (ds, m)
            if k not in CAL:
                continue
            byq = CAL[k]
            qs = list(byq)
            amin, n_star, k_star, all_in = alpha_scan([p for q in qs for p in byq[q]])
            n_all = sum(len(byq[q]) for q in qs)
            reps = []
            for _ in range(B):
                samp = [qs[rng.randrange(len(qs))] for _ in qs]
                reps.append(alpha_scan([p for q in samp for p in byq[q]])[0])
            lo, hi = ci(reps)
            cert = sum(v <= 0.10 for v in reps) / len(reps)
            print(f"  {ds:12s} {m:9s} alpha_min={amin:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"at n*={n_star}/{n_all}, k*={k_star}  accept-all={all_in:.3f}  "
                  f"gain over accept-all={all_in - amin:.3f}  "
                  f"certifies in {100 * cert:.0f}% of resamples")
    print("\nDone.")


if __name__ == "__main__":
    main()
