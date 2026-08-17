"""
Does the "always rewrite" fallback actually help or hurt, for the models
whose calibration failed and defaulted to tau_global=-1.0?

Ties directly to the project's original "Confidence Trap" hypothesis:
rewriting strips hedging language, which can make answers sound MORE
confident, which can then attract HARDER pushback in later rounds --
plausibly making things worse, not just "unproven better." Calibration
failing doesn't just mean "no certified guarantee" -- every response for
that model still gets rewritten unconditionally, and nobody has checked
whether that unconditional rewriting actually helps.

Uses ONLY the real, already-completed test-phase output (no new LLM
calls, no reruns): for every row where rewrite_triggered==True, compares
draft_sycophancy (the classification BEFORE the rewrite, i.e. what the
purified draft looked like) against sycophancy (the classification AFTER
the rewrite, i.e. the final delivered answer) -- both fields are already
stored per row by test_apply() in conformal_v2/run_conformal_v2.py.

Usage (from repo root, on the HPC, after the real suite has run):
    python3 analyze_rewrite_effect.py
"""
from __future__ import annotations
import json
import os

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
HS_DIR = "results/healthsearch_v6_1000"
MQ_DIR = "results/medDataset_v6_1000"

RUNS = []
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-on", f"{HS_DIR}/run_conformal_{key}_v6.json"))
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-off", f"{HS_DIR}/run_conformal_{key}_no_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-on", f"{MQ_DIR}/run_conformal_{key}_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-off", f"{MQ_DIR}/run_conformal_{key}_no_oracle_v6.json"))

# models we already know have a genuine, uncalibratable high fold rate
# (analyze_data_needed.py) -- these are the ones this question matters
# most for, but we report ALL models for comparison.
HIGH_FOLD_MODELS = {"llama_1b", "llama_3b", "gemma_4b"}


def load_records(out_path: str):
    if not os.path.exists(out_path):
        return None
    d = json.load(open(out_path))
    return d.get("records") or d.get("individual_records") or d.get("test_instances", [])


def rate(records, key):
    n = len(records)
    if n == 0:
        return dict(N=0, regressive=0.0, progressive=0.0, none=0.0)
    regr = sum(1 for r in records if r[key] == "regressive") / n
    prog = sum(1 for r in records if r[key] == "progressive") / n
    none = sum(1 for r in records if r[key] == "none") / n
    return dict(N=n, regressive=regr, progressive=prog, none=none)


def main():
    print(f"{'dataset':<12}{'model':<10}{'oracle':<11}{'n_rewritten':<13}"
          f"{'regr_before':<13}{'regr_after':<12}{'delta':<10}{'flag'}")
    print("-" * 100)

    totals = {"before": [], "after": []}

    for dataset, model, oracle, out_path in RUNS:
        records = load_records(out_path)
        if records is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}  MISSING: {out_path}")
            continue

        rewritten = [r for r in records if r.get("rewrite_triggered") is True]
        if not rewritten:
            print(f"{dataset:<12}{model:<10}{oracle:<11}{'0':<13}{'n/a':<13}{'n/a':<12}{'n/a':<10}(no rewrites triggered)")
            continue

        before = rate(rewritten, "draft_sycophancy")
        after = rate(rewritten, "sycophancy")
        delta = after["regressive"] - before["regressive"]

        flag = ""
        if model in HIGH_FOLD_MODELS:
            flag = "WORSE" if delta > 0.01 else ("BETTER" if delta < -0.01 else "~same")
            totals["before"].append(before["regressive"])
            totals["after"].append(after["regressive"])

        print(f"{dataset:<12}{model:<10}{oracle:<11}{before['N']:<13}"
              f"{before['regressive']:<13.3f}{after['regressive']:<12.3f}{delta:<+10.3f}{flag}")

    if totals["before"]:
        avg_before = sum(totals["before"]) / len(totals["before"])
        avg_after = sum(totals["after"]) / len(totals["after"])
        print("-" * 100)
        print(f"HIGH-FOLD-RATE MODELS ONLY (llama_1b, llama_3b, gemma_4b) -- "
              f"avg regressive rate among rewritten rows:")
        print(f"  before rewrite: {avg_before:.3f}")
        print(f"  after rewrite:  {avg_after:.3f}")
        print(f"  {'REWRITING MADE THINGS WORSE ON AVERAGE' if avg_after > avg_before else 'REWRITING HELPED ON AVERAGE'}")


if __name__ == "__main__":
    main()
