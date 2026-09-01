"""
Mentor request (2026-09, feedback quoted in the LM_corrections conversation
log): "compare directly against a rewrite-only baseline to see how much of
the sycophancy reduction is really coming from rewriting."

This is the zero-new-compute half of that ask. For every test row where
rewrite_triggered == True, compares draft_sycophancy (before the rewrite)
to sycophancy (after it) using data ALREADY collected in
run_conformal_<model>.json's individual_records -- no new LLM calls, no
reruns. Same method docs/REWRITE_POLICY_LOG.md used on the old (buggy,
pre-fix) v6 data; this repeats it on the current, corrected pipeline data
(folding-only labels, leak-free rewrite, deployable-only, exact-CRC
threshold) so the finding is checked against results that don't have any
known bugs behind them.

Important asymmetry to read correctly: for models whose calibration FAILS
(tau_global < 0), EVERY test draft gets rewritten (s > tau is true for any
s when tau=-1.0), so for those models "the conformal run" and "a rewrite-
everything policy" are numerically the same thing already -- this script's
output for those models directly answers the mentor's question with no
further experiment needed. For models that DO certify (tau_global >= 0,
e.g. Phi-1.5/2), rewrite is rare (only triggered on the few high-risk
items), so this script's rewrite_triggered_rate will be near zero for
them -- informative on its own (selective filtering is doing real work,
not defaulting to rewrite-everything), but doesn't tell you what would
happen if you rewrote everything for those models too. That second
question needs an actual new arm (--force_rewrite_all), not covered here.

Usage (from repo root, after checkpoints exist on disk):
    python3 analyze_rewrite_effect_v10.py
    python3 analyze_rewrite_effect_v10.py --dir results/v10_exact_crc_medquad
"""
from __future__ import annotations
import argparse
import json
import os

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
DEFAULT_DIR = "results/v9_night_medquad"


def load_records(path: str):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d.get("individual_records") or d.get("records") or d.get("test_instances", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default=DEFAULT_DIR)
    args = p.parse_args()

    print(f"results dir = {args.dir}\n")
    header = (f"{'model':<10}{'n_test':<8}{'rewrite_rate':<14}{'regr_before':<13}"
              f"{'regr_after':<12}{'delta':<9}verdict")
    print(header)
    print("-" * len(header))

    for model in MODELS:
        path = os.path.join(args.dir, f"run_conformal_{model}.json")
        recs = load_records(path)
        if not recs:
            print(f"{model:<10}  MISSING ({path})")
            continue

        n_test = len(recs)
        rewritten = [r for r in recs if r.get("rewrite_triggered")]
        n_rw = len(rewritten)
        rewrite_rate = n_rw / n_test if n_test else 0.0

        if n_rw == 0:
            print(f"{model:<10}{n_test:<8}{rewrite_rate:<14.3f}"
                  f"{'n/a':<13}{'n/a':<12}{'n/a':<9}no rewrites triggered -- nothing to compare")
            continue

        regr_before = sum(1 for r in rewritten if r.get("draft_sycophancy") == "regressive") / n_rw
        regr_after = sum(1 for r in rewritten if r.get("sycophancy") == "regressive") / n_rw
        delta = regr_after - regr_before

        if rewrite_rate > 0.95:
            scope_note = "rewrite_rate~1.0 -> this IS the rewrite-everything policy already"
        else:
            scope_note = "selective (partial rewrite) -- not equivalent to rewrite-everything"

        if delta < -0.02:
            verdict = f"rewrite HELPS ({scope_note})"
        elif delta > 0.02:
            verdict = f"rewrite HURTS ({scope_note})"
        else:
            verdict = f"roughly neutral ({scope_note})"

        print(f"{model:<10}{n_test:<8}{rewrite_rate:<14.3f}{regr_before:<13.3f}"
              f"{regr_after:<12.3f}{delta:<+9.3f}{verdict}")


if __name__ == "__main__":
    main()
