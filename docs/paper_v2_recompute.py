#!/usr/bin/env python3
"""
Independent re-derivation of every table and in-text statistic in sigconf.tex
from the raw per-item result files. Written for docs/PAPER_REVIEW_V2.md
(second-pass review). Deliberately shares no code with analyze_*.py or
hpc_verify_paper.py: all rates, denominators, pairing, and the exact-CRC scan
are re-implemented here from the record schema.

Run on the HPC from the repo root (no LLM calls, no writes except the optional
--json dump):

    python3 docs/paper_v2_recompute.py
    python3 docs/paper_v2_recompute.py --json /tmp/paper_v2_recompute.json

Inputs (all read-only):
  results/final_n1000_{medquad,healthsearch}/run_baseline_<m>.json
  results/final_n1000_{medquad,healthsearch}/run_conformal_<m>.json
  results/final_n1000_{medquad,healthsearch}/thresholds_<m>.json
  results/oracle_n1000_{medquad,healthsearch}/thresholds_<m>.json   (Table 5)
  results/oracle_n1000_{medquad,healthsearch}/calib_<m>.json         (optional)

Record schemas used (verified against run_eval.py:174-185,
run_conformal_v2.py:484-500 and :736-764):
  baseline  individual_records: idx, where, strength, first_label, after_label,
            sycophancy, question
  CCR test  individual_records: idx, where, strength, first_label, draft_label,
            draft_sycophancy, risk_score, tau, rewrite_triggered, after_label,
            sycophancy, question, draft_claims_total/kept/dropped,
            final_claims_total/kept/dropped, draft_answer_raw, draft_answer,
            final_answer
  CCR calib calibration_debug_records: idx, where, strength, first_label,
            draft_label, draft_sycophancy, bad, risk_score, claims_total, ...

Every paper number is compared at the paper's displayed precision. A row is
PASS when the recomputed value rounds (half-up) to the printed value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

MODELS = ["gemma_1b", "gemma_4b", "llama_1b", "llama_3b", "phi_1.5", "phi_2"]
DATASETS = ["medquad", "healthsearch"]
WHERE = ["in-context", "preemptive"]
STRENGTHS = ["simple", "ethos", "justification", "citation"]

# --------------------------------------------------------------------------
# Paper values (sigconf.tex @ commit 24c6bec), by table / line, for PASS/FAIL
# --------------------------------------------------------------------------
# Table 1 (l.788-793): model -> (MQ regr, MQ prog, HSQA regr, HSQA prog)
PAPER_T1 = {
    "gemma_1b": (20.0, 30.8, 40.4, 18.7),
    "gemma_4b": (39.8, 31.2, 64.7, 11.8),
    "llama_1b": (14.8, 13.8, 23.3, 8.1),
    "llama_3b": (33.8, 25.0, 48.7, 11.7),
    "phi_1.5": (6.2, 18.5, 13.8, 18.4),
    "phi_2": (8.1, 38.2, 14.5, 40.3),
}
# Table 2 (l.859-864): model -> (MQ alpha_min, HSQA alpha_min); rewrite 100%/0%
PAPER_T2 = {
    "gemma_1b": (0.124, 0.187), "gemma_4b": (0.184, 0.339),
    "llama_1b": (0.120, 0.187), "llama_3b": (0.150, 0.269),
    "phi_1.5": (0.023, 0.081), "phi_2": (0.048, 0.035),
}
# Table 3 (l.914-919): model -> (MQ dRegr, MQ dProg, HSQA dRegr, HSQA dProg)
PAPER_T3 = {
    "gemma_1b": (-2.1, -11.9, -4.4, -7.6),
    "gemma_4b": (-10.2, -6.1, -21.1, -0.4),
    "llama_1b": (3.0, -5.9, 6.3, -1.2),
    "llama_3b": (0.0, -7.4, -3.5, -3.2),
    "phi_1.5": (-1.5, 10.9, -4.0, 12.0),
    "phi_2": (-3.4, 0.5, -6.1, 3.7),
}
# Table 4 (l.958-967): (dataset, model) -> (before, after, delta)
PAPER_T4 = {
    ("medquad", "gemma_1b"): (12.9, 17.9, 5.0), ("medquad", "gemma_4b"): (28.1, 29.6, 1.5),
    ("medquad", "llama_1b"): (13.4, 17.8, 4.4), ("medquad", "llama_3b"): (28.5, 33.8, 5.2),
    ("healthsearch", "gemma_1b"): (27.1, 36.0, 8.9), ("healthsearch", "gemma_4b"): (41.8, 43.7, 1.8),
    ("healthsearch", "llama_1b"): (22.0, 29.6, 7.6), ("healthsearch", "llama_3b"): (35.4, 45.2, 9.8),
}
# Table 5 (l.1091-1096): model -> (MQ oracle, HSQA oracle)
PAPER_T5_ORACLE = {
    "gemma_1b": (0.079, 0.168), "gemma_4b": (0.201, 0.308),
    "llama_1b": (0.090, 0.177), "llama_3b": (0.145, 0.251),
    "phi_1.5": (0.026, 0.066), "phi_2": (0.042, 0.028),
}
# In-text (l.869-871): accepted-set regressive rate for certified models
PAPER_ACCEPTED_REGR = {("medquad", "phi_1.5"): 4.7, ("medquad", "phi_2"): 4.7,
                       ("healthsearch", "phi_1.5"): 9.8, ("healthsearch", "phi_2"): 8.4}


def r1(x: float) -> float:
    """Round half-up to 1 decimal (the convention a human uses when typing a table)."""
    return float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def r3(x: float) -> float:
    return float(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def pf(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def qhash(q: str) -> str:
    # Must match conformal_v2/run_conformal_v2.py:_question_hash (sha1, strip, 16 hex).
    return hashlib.sha1(q.strip().encode("utf-8")).hexdigest()[:16]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def records(d, key="individual_records"):
    if d is None:
        return None
    for k in (key, "records", "test_instances"):
        if isinstance(d.get(k), list):
            return d[k]
    return None


def share(rows, pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else float("nan")


# --------------------------------------------------------------------------
# Exact-CRC scan, re-implemented from eq. (crc) / eq. (alpha_min) in the paper
# --------------------------------------------------------------------------
def crc_scan(scores, bad):
    """Return (tau_star, alpha_min, n_at_min, k_at_min, table) where table is a
    list of (tau, n, k, bound) over every distinct score value, ascending."""
    pairs = sorted(zip(scores, bad))
    table = []
    n = k = 0
    i = 0
    while i < len(pairs):
        tau = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == tau:
            n += 1
            k += pairs[i][1]
            i += 1
        table.append((tau, n, k, (k + 1) / (n + 1)))
    amin = min(t[3] for t in table)
    rows_at_min = [t for t in table if t[3] == amin]
    tau_star, n_min, k_min, _ = rows_at_min[0]
    return tau_star, amin, n_min, k_min, table


def crc_fit(table, alpha):
    """Most permissive tau whose bound <= alpha, else -1.0 (paper l.567, 604-607)."""
    feas = [t for t in table if t[3] <= alpha]
    return feas[-1][0] if feas else -1.0


# --------------------------------------------------------------------------
# Section A: Tables 1 + 3 + conditional statistics + erroneous
# --------------------------------------------------------------------------
def section_a(root, ds, out):
    D = f"{root}/final_n1000_{ds}"
    col = 0 if ds == "medquad" else 2
    print(f"\n=== A. {ds}: Table 1 (baseline, paired to CCR test items) and Table 3 (CCR - baseline) ===")
    print(f"{'model':9s} {'baseN':>5s} {'regr':>6s} {'prog':>6s} {'err0':>5s} {'errA':>5s} | "
          f"{'ccrN':>5s} {'regr':>6s} {'prog':>6s} {'err0':>5s} {'errF':>5s} | "
          f"{'dRegr':>6s} {'dProg':>6s} | T1 T3")
    for m in MODELS:
        th = load_json(f"{D}/thresholds_{m}.json")
        b = records(load_json(f"{D}/run_baseline_{m}.json"))
        c = records(load_json(f"{D}/run_conformal_{m}.json"))
        if th is None or b is None or c is None:
            print(f"{m:9s} MISSING file(s)")
            continue
        calib = set(th["data_split"]["calib_question_hashes"])
        bp = [r for r in b if qhash(r["question"]) not in calib]
        n_b, n_c = len(bp), len(c)
        br = 100 * share(bp, lambda r: r["sycophancy"] == "regressive")
        bg = 100 * share(bp, lambda r: r["sycophancy"] == "progressive")
        cr = 100 * share(c, lambda r: r["sycophancy"] == "regressive")
        cg = 100 * share(c, lambda r: r["sycophancy"] == "progressive")
        be0 = 100 * share(bp, lambda r: r["first_label"] == "erroneous")
        bea = 100 * share(bp, lambda r: r["after_label"] == "erroneous")
        ce0 = 100 * share(c, lambda r: r["first_label"] == "erroneous")
        cef = 100 * share(c, lambda r: r["after_label"] == "erroneous")
        dr, dg = cr - br, cg - bg
        t1 = PAPER_T1[m]
        t3 = PAPER_T3[m]
        ok1 = (r1(br) == t1[col]) and (r1(bg) == t1[col + 1])
        ok3 = (r1(dr) == t3[col]) and (r1(dg) == t3[col + 1])
        print(f"{m:9s} {n_b:5d} {br:6.2f} {bg:6.2f} {be0:5.1f} {bea:5.1f} | "
              f"{n_c:5d} {cr:6.2f} {cg:6.2f} {ce0:5.1f} {cef:5.1f} | "
              f"{dr:+6.2f} {dg:+6.2f} | {pf(ok1)} {pf(ok3)}"
              f"   paper T1=({t1[col]},{t1[col+1]}) T3=({t3[col]:+},{t3[col+1]:+})")
        # conditional stats (l.765-772, 803-808, 836-840)
        c0 = [r for r in bp if r["first_label"] == "correct"]
        i0 = [r for r in bp if r["first_label"] == "incorrect"]
        init = len(c0) / n_b
        cap = share(c0, lambda r: r["sycophancy"] == "regressive")
        selfc = share(i0, lambda r: r["sycophancy"] == "progressive")
        # same, CCR arm (initial answers regenerated in that arm)
        cc0 = [r for r in c if r["first_label"] == "correct"]
        cinit = len(cc0) / n_c
        print(f"{'':9s}   baseline: init_correct={init:.4f} capit|correct={cap:.4f} (n={len(c0)}) "
              f"selfcorr|incorrect={selfc:.4f} (n={len(i0)}) | CCR arm init_correct={cinit:.4f}")
        out.setdefault("A", {})[(ds, m)] = dict(
            baseN=n_b, ccrN=n_c, base_regr=br, base_prog=bg, ccr_regr=cr, ccr_prog=cg,
            d_regr=dr, d_prog=dg, init_correct=init, capit_given_correct=cap,
            selfcorr_given_incorrect=selfc, ccr_init_correct=cinit,
            base_err_first=be0, base_err_after=bea, ccr_err_first=ce0, ccr_err_final=cef,
            n_first_correct=len(c0), n_first_incorrect=len(i0))
        if ds == "medquad":
            # baseline file holds all 1,000 questions: pairing removes the 200 calibration
            # questions (1,600 rows) plus the 5 test slots whose text duplicates a
            # calibration question (40 rows) -> 8000 - 1640 = 6360 (paper l.750, l.1177-1181)
            print(f"{'':9s}   baseline rows: {len(b)} total, {len(b) - n_b} dropped by hash pairing "
                  f"(expect 1600 calib + 40 duplicate-text = 1640), paired N={n_b} (paper: 6,360) "
                  f"{pf(len(b) - n_b == 1640 and n_b == 6360)}")


# --------------------------------------------------------------------------
# Section B: Table 2 — alpha_min, tau, rewrite rate re-derived from calibration
# --------------------------------------------------------------------------
def section_b(root, ds, out):
    D = f"{root}/final_n1000_{ds}"
    col = 0 if ds == "medquad" else 1
    print(f"\n=== B. {ds}: Table 2 (exact-CRC scan re-implemented from calibration_debug_records) ===")
    for m in MODELS:
        th = load_json(f"{D}/thresholds_{m}.json")
        d = load_json(f"{D}/run_conformal_{m}.json")
        cal = records(d, "calibration_debug_records") if d else None
        if th is None:
            print(f"{m:9s} MISSING thresholds")
            continue
        file_amin = th["alpha_min"]
        fr = Fraction(file_amin).limit_denominator(1601)
        line = (f"{m:9s} file: tau={th['tau_global']} alpha_min={file_amin:.6f} "
                f"(= {fr.numerator}/{fr.denominator} or a multiple) xi={th['xi_certified']} "
                f"paper={PAPER_T2[m][col]} {pf(r3(file_amin) == PAPER_T2[m][col])}")
        if not cal:
            print(line + "  [no calibration_debug_records in run_conformal json]")
            continue
        scores = [float(r["risk_score"]) for r in cal]
        bad = [int(r["bad"]) for r in cal]
        tau_star, amin, n_min, k_min, table = crc_scan(scores, bad)
        tau_fit = crc_fit(table, float(th["alpha"]))
        n_q = len({r["idx"] for r in cal})
        print(line)
        print(f"{'':9s} recomputed: n_calib={len(scores)} (questions={n_q}) distinct_scores={len(table)} "
              f"alpha_min={amin:.6f} at tau*={tau_star} with n={n_min}, k={k_min} "
              f"| tau_fit={tau_fit} {pf(abs(amin - file_amin) < 1e-12 and tau_fit == th['tau_global'])}")
        print(f"{'':9s} risk_score==1.0 share (upper bound on parse failures): "
              f"{100*share(cal, lambda r: float(r['risk_score']) == 1.0):.1f}% | "
              f"bad rate overall={sum(bad)/len(bad):.4f} | "
              f"calib erroneous draft labels={100*share(cal, lambda r: r['draft_label']=='erroneous'):.1f}% "
              f"| claims_total==0: {100*share(cal, lambda r: r.get('claims_total',1)==0):.1f}%")
        print(f"{'':9s} scan (tau, n_accepted, k_bad, (k+1)/(n+1)):")
        for t in table:
            flag = " <-- alpha_min" if t[3] == amin else ""
            print(f"{'':13s} {t[0]:<5} n={t[1]:<5d} k={t[2]:<4d} bound={t[3]:.4f}{flag}")
        out.setdefault("B", {})[(ds, m)] = dict(
            file_alpha_min=file_amin, recomputed_alpha_min=amin, tau_star=tau_star,
            n_at_min=n_min, k_at_min=k_min, tau_fit=tau_fit, n_calib=len(scores),
            distinct_scores=len(table), scan=table)


# --------------------------------------------------------------------------
# Section C: Table 4 — rewrite effect, and identity checks
# --------------------------------------------------------------------------
def section_c(root, ds, out):
    D = f"{root}/final_n1000_{ds}"
    print(f"\n=== C. {ds}: Table 4 (rows where rewrite fired) and identities ===")
    for m in MODELS:
        c = records(load_json(f"{D}/run_conformal_{m}.json"))
        if c is None:
            print(f"{m:9s} MISSING")
            continue
        rw = [r for r in c if r.get("rewrite_triggered")]
        rate_rw = len(rw) / len(c)
        exp_rw = 1.0 if m.startswith(("gemma", "llama")) else 0.0
        line = f"{m:9s} rewrite_rate={rate_rw:.4f} (Table 2 expects {exp_rw:.0%}) {pf(rate_rw == exp_rw)}"
        if not rw:
            # certified models: accepted set == full test set; realized regressive rate (l.869-873)
            acc = [r for r in c if float(r["risk_score"]) <= float(r["tau"])]
            ar = 100 * share(acc, lambda r: r["sycophancy"] == "regressive")
            exp = PAPER_ACCEPTED_REGR.get((ds, m))
            print(line + f" | accepted n={len(acc)} regressive among accepted={ar:.2f}% "
                  f"paper={exp} {pf(exp is not None and r1(ar) == exp)}")
            out.setdefault("C", {})[(ds, m)] = dict(rewrite_rate=rate_rw, accepted_regr=ar, accepted_n=len(acc))
            continue
        before = 100 * share(rw, lambda r: r["draft_sycophancy"] == "regressive")
        after = 100 * share(rw, lambda r: r["sycophancy"] == "regressive")
        delta = after - before
        exp = PAPER_T4[(ds, m)]
        ok = (r1(before), r1(after), r1(delta)) == exp
        print(line + f" | before={before:.2f} after={after:.2f} delta={delta:+.2f} paper={exp} {pf(ok)}")
        # draft-level transitions through the rewrite
        trans = Counter((r["draft_label"], r["after_label"]) for r in rw)
        print(f"{'':9s} draft_label -> after_label counts: {dict(sorted(trans.items()))}")
        # final-answer fallback / empty detection
        nvc = 100 * share(c, lambda r: r.get("final_answer", "") == "(No valid claims found)")
        nvd = 100 * share(c, lambda r: r.get("draft_answer", "") == "(No valid claims found)")
        print(f"{'':9s} '(No valid claims found)' share: draft={nvd:.2f}% final={nvc:.2f}% ; "
              f"draft_claims_total==0: {100*share(c, lambda r: r.get('draft_claims_total',1)==0):.2f}%")
        out.setdefault("C", {})[(ds, m)] = dict(rewrite_rate=rate_rw, before=before, after=after, delta=delta)


# --------------------------------------------------------------------------
# Section D: Table 5 oracle alpha_min
# --------------------------------------------------------------------------
def section_d(root, out):
    print("\n=== D. Table 5: oracle alpha_min (results/oracle_n1000_*/thresholds_*.json) ===")
    for ds in DATASETS:
        col = 0 if ds == "medquad" else 1
        D = f"{root}/oracle_n1000_{ds}"
        dep = f"{root}/final_n1000_{ds}"
        for m in MODELS:
            th = load_json(f"{D}/thresholds_{m}.json")
            thd = load_json(f"{dep}/thresholds_{m}.json")
            if th is None:
                print(f"{ds:12s} {m:9s} MISSING")
                continue
            same_calib = (thd is not None and th["data_split"]["calib_question_hashes"]
                          == thd["data_split"]["calib_question_hashes"])
            exp = PAPER_T5_ORACLE[m][col]
            cal = records(load_json(f"{D}/calib_{m}.json"), "calibration_debug_records")
            extra = ""
            if cal:
                sc = [float(r["risk_score"]) for r in cal]
                bd = [int(r["bad"]) for r in cal]
                _, amin, n_min, k_min, table = crc_scan(sc, bd)
                extra = (f" | recomputed alpha_min={amin:.4f} (n={n_min},k={k_min}, distinct={len(table)}) "
                         f"{pf(abs(amin - th['alpha_min']) < 1e-12)}")
            print(f"{ds:12s} {m:9s} alpha_min={th['alpha_min']:.4f} tau={th['tau_global']} "
                  f"paper={exp} {pf(r3(th['alpha_min']) == exp)} | same 200 calib hashes as deployable: "
                  f"{pf(same_calib)}{extra}")
            out.setdefault("D", {})[(ds, m)] = dict(oracle_alpha_min=th["alpha_min"], same_calib=same_calib)


# --------------------------------------------------------------------------
# Section E: qualifier checks (every "all/every/none/only" sentence)
# --------------------------------------------------------------------------
def section_e(out):
    A = out.get("A", {})
    if len(A) < 12:
        print("\n=== E. qualifier checks skipped (missing files) ===")
        return
    print("\n=== E. Qualifier checks (literal truth of all/every/none/only statements) ===")
    g = lambda ds, m, k: A[(ds, m)][k]
    checks = []
    checks.append(("l.760-761 regressive higher on HSQA for all six",
                   all(g("healthsearch", m, "base_regr") > g("medquad", m, "base_regr") for m in MODELS)))
    lower = [m for m in MODELS if g("healthsearch", m, "base_prog") < g("medquad", m, "base_prog")]
    checks.append(("l.761-762 progressive lower on HSQA for five of six (all but Phi-2)",
                   sorted(lower) == sorted([m for m in MODELS if m != "phi_2"])))
    checks.append(("l.765 initial accuracy higher on HSQA for every model",
                   all(g("healthsearch", m, "init_correct") > g("medquad", m, "init_correct") for m in MODELS)))
    gaps = {m: abs(g("healthsearch", m, "capit_given_correct") - g("medquad", m, "capit_given_correct")) for m in MODELS}
    checks.append((f"l.766-767 capitulation|correct within ~6 points (max gap={100*max(gaps.values()):.1f})",
                   max(gaps.values()) <= 0.065))
    checks.append(("l.771-772 self-correction|incorrect higher on HSQA for every model",
                   all(g("healthsearch", m, "selfcorr_given_incorrect") > g("medquad", m, "selfcorr_given_incorrect") for m in MODELS)))
    fam = [("gemma_4b", "gemma_1b"), ("llama_3b", "llama_1b"), ("phi_2", "phi_1.5")]
    checks.append(("l.798-801 larger model more regressive within every family on both datasets",
                   all(g(ds, big, "base_regr") > g(ds, small, "base_regr") for ds in DATASETS for big, small in fam)))
    checks.append(("l.803 same ordering conditioned on correct initial answer",
                   all(g(ds, big, "capit_given_correct") > g(ds, small, "capit_given_correct") for ds in DATASETS for big, small in fam)))
    phi_init = [g(ds, m, "init_correct") for ds in DATASETS for m in ("phi_1.5", "phi_2")]
    phi_cap = [g(ds, m, "capit_given_correct") for ds in DATASETS for m in ("phi_1.5", "phi_2")]
    checks.append((f"l.806-808 Phi initially correct 9-19% (got {100*min(phi_init):.1f}-{100*max(phi_init):.1f}) "
                   f"and capitulate 71-79% (got {100*min(phi_cap):.1f}-{100*max(phi_cap):.1f})",
                   0.085 <= min(phi_init) and max(phi_init) < 0.195 and 0.705 <= min(phi_cap) and max(phi_cap) < 0.795))
    n_mq = sum(1 for m in MODELS if g("medquad", m, "d_regr") < 0)
    n_hs = sum(1 for m in MODELS if g("healthsearch", m, "d_regr") < 0)
    checks.append((f"l.893-895 CCR lowers regressive for 4/6 MQ and 5/6 HSQA (got {n_mq}/6, {n_hs}/6; "
                   f"llama_3b MQ dRegr={g('medquad','llama_3b','d_regr'):+.3f})", (n_mq, n_hs) == (4, 5)))
    checks.append(("l.895-896 no model's dRegr changes sign between datasets (treating |d|<0.05 as 0)",
                   all((g("medquad", m, "d_regr") >= -0.05) == (g("healthsearch", m, "d_regr") >= -0.05)
                       or abs(g("medquad", m, "d_regr")) < 0.05 for m in MODELS)))
    checks.append(("l.896 |dRegr| larger on HSQA for every model",
                   all(abs(g("healthsearch", m, "d_regr")) > abs(g("medquad", m, "d_regr")) for m in MODELS)))
    checks.append(("l.898 LLaMA-1B the only model made worse (dRegr>0) on both datasets",
                   [m for m in MODELS if g("medquad", m, "d_regr") > 0 and g("healthsearch", m, "d_regr") > 0] == ["llama_1b"]))
    unc = ["gemma_1b", "gemma_4b", "llama_1b", "llama_3b"]
    checks.append(("l.926-928 progressive falls on both datasets for the four uncertified",
                   all(g(ds, m, "d_prog") < 0 for ds in DATASETS for m in unc)))
    checks.append(("l.929-930 progressive rises on both datasets for the two certified",
                   all(g(ds, m, "d_prog") > 0 for ds in DATASETS for m in ("phi_1.5", "phi_2"))))
    C = out.get("C", {})
    if len(C) >= 8:
        d4 = [C[(ds, m)]["delta"] for ds in DATASETS for m in unc]
        checks.append(("l.972-973 rewrite increases regressive in all eight pairs", all(x > 0 for x in d4)))
        checks.append(("l.1149 LLaMA-3.2 1B 'the smallest model evaluated' (FALSE by parameter count: "
                       "Gemma-3 1B ~1.0B < LLaMA-3.2 1B ~1.24B) -- not a data check", False))
    for name, ok in checks:
        print(f"  [{pf(ok)}] {name}")
    out["E"] = checks


# --------------------------------------------------------------------------
# Section F: arm-prompt confound (run_eval.py:98 vs run_conformal_v2.py:224)
# --------------------------------------------------------------------------
def section_f(root, ds, out):
    D = f"{root}/final_n1000_{ds}"
    print(f"\n=== F. {ds}: baseline-vs-CCR by draft slot (prompts identical for preemptive and in-context/simple; "
          f"chained in-context slots differ between arms) ===")
    print(f"{'model':9s} {'slot':24s} {'baseRegr':>8s} {'ccrRegr':>8s} {'dRegr':>7s} | {'baseProg':>8s} {'ccrProg':>8s} {'dProg':>7s}")
    for m in MODELS:
        th = load_json(f"{D}/thresholds_{m}.json")
        b = records(load_json(f"{D}/run_baseline_{m}.json"))
        c = records(load_json(f"{D}/run_conformal_{m}.json"))
        if th is None or b is None or c is None:
            continue
        calib = set(th["data_split"]["calib_question_hashes"])
        bp = [r for r in b if qhash(r["question"]) not in calib]
        groups = [("preemptive (all 4)", lambda r: r["where"] == "preemptive"),
                  ("in-context/simple", lambda r: r["where"] == "in-context" and r["strength"] == "simple"),
                  ("in-context/chained x3", lambda r: r["where"] == "in-context" and r["strength"] != "simple")]
        for name, sel in groups:
            bb = [r for r in bp if sel(r)]
            cc = [r for r in c if sel(r)]
            br = 100 * share(bb, lambda r: r["sycophancy"] == "regressive")
            cr = 100 * share(cc, lambda r: r["sycophancy"] == "regressive")
            bg = 100 * share(bb, lambda r: r["sycophancy"] == "progressive")
            cg = 100 * share(cc, lambda r: r["sycophancy"] == "progressive")
            print(f"{m:9s} {name:24s} {br:8.2f} {cr:8.2f} {cr-br:+7.2f} | {bg:8.2f} {cg:8.2f} {cg-bg:+7.2f}")
            out.setdefault("F", {})[(ds, m, name)] = dict(base_regr=br, ccr_regr=cr, base_prog=bg, ccr_prog=cg)
        # first_label agreement between arms (initial answers are regenerated per arm)
        bq = defaultdict(set)
        for r in bp:
            bq[qhash(r["question"])].add(r["first_label"])
        cq = defaultdict(set)
        for r in c:
            cq[qhash(r["question"])].add(r["first_label"])
        common = [h for h in cq if h in bq and len(bq[h]) == 1 and len(cq[h]) == 1]
        agree = sum(1 for h in common if bq[h] == cq[h])
        print(f"{'':9s} first_label agreement baseline vs CCR on {len(common)} unambiguous shared questions: "
              f"{agree} ({100*agree/max(1,len(common)):.1f}%)")
        out.setdefault("F_agree", {})[(ds, m)] = (agree, len(common))


# --------------------------------------------------------------------------
# Section G: duplicates and transition matrices
# --------------------------------------------------------------------------
def section_g(root, ds, out):
    D = f"{root}/final_n1000_{ds}"
    print(f"\n=== G. {ds}: duplicate question texts and 3x3 transition matrices ===")
    for m in MODELS:
        c = records(load_json(f"{D}/run_conformal_{m}.json"))
        if c is None:
            continue
        hs = Counter(qhash(r["question"]) for r in c)
        dup_slots = sum(v // 8 - 1 for v in hs.values() if v > 8)
        labs = ["correct", "incorrect", "erroneous"]
        t_fd = Counter((r["first_label"], r["draft_label"]) for r in c)
        t_ff = Counter((r["first_label"], r["after_label"]) for r in c)
        print(f"{m:9s} CCR test: {len(hs)} distinct question texts over {len(c)//8} slots "
              f"({dup_slots} duplicate slots)")
        print(f"{'':9s} first->filtered-draft: " + "  ".join(f"{a[:3]}->{b[:3]}={t_fd[(a,b)]}" for a in labs for b in labs))
        print(f"{'':9s} first->final:          " + "  ".join(f"{a[:3]}->{b[:3]}={t_ff[(a,b)]}" for a in labs for b in labs))
        out.setdefault("G", {})[(ds, m)] = dict(first_draft=dict(t_fd), first_final=dict(t_ff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--json", default=None, help="optional path to dump all recomputed numbers")
    args = ap.parse_args()
    out = {}
    for ds in DATASETS:
        section_a(args.root, ds, out)
    for ds in DATASETS:
        section_b(args.root, ds, out)
    for ds in DATASETS:
        section_c(args.root, ds, out)
    section_d(args.root, out)
    section_e(out)
    for ds in DATASETS:
        section_f(args.root, ds, out)
    for ds in DATASETS:
        section_g(args.root, ds, out)
    if args.json:
        def keyfix(o):
            if isinstance(o, dict):
                return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): keyfix(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [keyfix(v) for v in o]
            return o
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(keyfix(out), f, indent=1, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
