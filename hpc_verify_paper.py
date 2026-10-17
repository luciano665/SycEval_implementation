"""
One-shot verification of the HPC-only numbers cited in sigconf.tex.

Companion to docs/PAPER_REVIEW.md section 4. Run on the HPC from the
repo root (syceval env) after the final N=1000 suite has finished:

    python3 hpc_verify_paper.py

Reads results/final_n1000_{medquad,healthsearch}/ and
results/oracle_n1000_*/ ; no LLM calls, no writes.
"""
from __future__ import annotations
import glob
import json
import os
import sys

sys.path.insert(0, ".")
from conformal_v2.run_conformal_v2 import _question_hash  # noqa: E402

MODELS = ["gemma_1b", "gemma_4b", "llama_1b", "llama_3b", "phi_1.5", "phi_2"]


def recs(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return (d.get("records") or d.get("individual_records")
            or d.get("test_instances", []))


def rate(rows, key, val):
    return sum(1 for r in rows if r.get(key) == val) / max(1, len(rows))


def paired_baseline(D, m):
    th = json.load(open(f"{D}/thresholds_{m}.json"))
    cal = set(th["data_split"]["calib_question_hashes"])
    b = recs(f"{D}/run_baseline_{m}.json")
    if b is None:
        return None
    return [r for r in b if _question_hash(r["question"]) not in cal]


def section_a(D):
    print("A) TABLES 1+3 (paired baseline vs CCR)")
    print("   model     | baseN  regr  prog | ccrN   regr  prog | dRegr dProg")
    for m in MODELS:
        bp = paired_baseline(D, m)
        c = recs(f"{D}/run_conformal_{m}.json")
        if bp is None or c is None:
            print(f"   {m:<9} | MISSING file")
            continue
        br = rate(bp, "sycophancy", "regressive")
        bg = rate(bp, "sycophancy", "progressive")
        cr = rate(c, "sycophancy", "regressive")
        cg = rate(c, "sycophancy", "progressive")
        print(f"   {m:<9} | {len(bp):<5} {br*100:5.1f} {bg*100:5.1f} "
              f"| {len(c):<5} {cr*100:5.1f} {cg*100:5.1f} "
              f"| {(cr-br)*100:+5.1f} {(cg-bg)*100:+5.1f}")


def section_b(D):
    print("B) TABLE 4 (rows where rewrite fired)")
    print("   model     | rw_rate | regr_before regr_after delta")
    for m in MODELS:
        c = recs(f"{D}/run_conformal_{m}.json")
        if c is None:
            continue
        rw = [r for r in c if r.get("rewrite_triggered")]
        if not rw:
            print(f"   {m:<9} | 0.000   | (no rewrites)")
            continue
        a = rate(rw, "draft_sycophancy", "regressive")
        z = rate(rw, "sycophancy", "regressive")
        print(f"   {m:<9} | {len(rw)/len(c):.3f}   | {a*100:5.1f} "
              f"{z*100:5.1f} {(z-a)*100:+5.1f}")


def section_d(D):
    print("D) CONDITIONAL STATS (paired baseline)")
    print("   model     | init_correct | capit|correct | selfcorr|incorrect")
    for m in MODELS:
        bp = paired_baseline(D, m)
        if bp is None:
            continue
        c0 = [r for r in bp if r["first_label"] == "correct"]
        i0 = [r for r in bp if r["first_label"] == "incorrect"]
        print(f"   {m:<9} | {len(c0)/len(bp):.3f}        "
              f"| {rate(c0, 'sycophancy', 'regressive'):.3f}         "
              f"| {rate(i0, 'sycophancy', 'progressive'):.3f}")


def section_c():
    print("\n######## C) ORACLE alpha_min (raw files) ########")
    for f in sorted(glob.glob("results/oracle_n1000_*/thresholds_*.json")):
        d = json.load(open(f))
        name = f.split("results/")[1]
        print(f"   {name:<48} alpha_min={d['alpha_min']:.3f} "
              f"tau={d['tau_global']}")


def section_e():
    print("\n######## E) FIGURE 1 CASE (llama_3b HSQA, von Willebrand) ########")
    c = recs("results/final_n1000_healthsearch/run_conformal_llama_3b.json")
    c = c or []
    hits = [r for r in c if "willebrand" in r.get("question", "").lower()]
    print("   matching rows:", len(hits))
    if hits:
        print("   keys:", sorted(hits[0].keys()))
    for r in hits:
        if r.get("strength") != "simple":
            continue
        scalars = {}
        for k, v in r.items():
            if k in ("question", "truth"):
                continue
            if isinstance(v, (list, dict)):
                continue
            if len(str(v)) < 60:
                scalars[k] = v
        print("   scalars:", scalars)
        for k, v in r.items():
            if isinstance(v, list) and "claim" in k.lower():
                print("   ", k, "=", v)


def section_f():
    print("\n######## F) SCORER DIAGNOSTICS ########")
    for s in ["analyze_v3_verify.py", "analyze_gptoss_verify.py"]:
        print(f"--- {s} ---")
        os.system(f"python3 {s} 2>&1 | tail -8")


def section_g():
    print("\n######## G) RERUN NOISE (oracle logs) ########")
    cmd = ("grep -hoE 'alpha_min[^,}]*' logs/error_oracle1k* 2>/dev/null "
           "| sort | uniq -c | head -40")
    if os.system(cmd) != 0:
        print("   (no logs/error_oracle1k* found)")


def main():
    for ds in ["medquad", "healthsearch"]:
        D = f"results/final_n1000_{ds}"
        print(f"\n######## {ds.upper()} ########")
        section_a(D)
        section_b(D)
        section_d(D)
    section_c()
    section_e()
    section_f()
    section_g()


if __name__ == "__main__":
    main()
