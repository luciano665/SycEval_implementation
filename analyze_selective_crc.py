"""
Retrofit the exact-CRC bound and the selective-certification diagnostic
(conformal_v2/conformal_thresholds.py: fit_global_threshold_exact_crc,
certifiable_fraction) onto ALREADY-COLLECTED calibration checkpoint data --
no new LLM calls, no reruns. Same pattern as analyze_data_needed.py.

What this answers, per model:
  1. Does the exact CRC bound (Angelopoulos & Bates, provably valid under
     the threshold scan -- see conformal_thresholds.py docstrings) certify
     where the current Wilson-based fit_global_threshold doesn't? If so,
     that specific failure was a bound-choice artifact, not a real ceiling.
  2. alpha_min: the loosest alpha at which this model has ANY exact-CRC
     certifiable region at all. Replaces the "we swept alpha up to 0.25
     and it still failed" grid search with one exact number per model.

Points at the v9 (deployable-only, folding-only-labels) suite by default,
the current canonical one -- not the older, bugged v6 results. Override
with --dir to point at a different results directory (e.g. a synthetic
fixture, or v9_judge_split_medquad once that suite lands).

Usage (from repo root, after checkpoints exist on disk):
    python3 analyze_selective_crc.py --alpha 0.10
    python3 analyze_selective_crc.py --alpha 0.10 --dir /path/to/fixture
"""
from __future__ import annotations
import argparse
import json
import os

from conformal_v2.conformal_thresholds import (
    certifiable_fraction,
    fit_global_threshold,
    fit_global_threshold_exact_crc,
)

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
DEFAULT_DIR = "results/v9_night_medquad"


def load_calib_records(results_dir: str, model: str):
    ckpt = os.path.join(results_dir, f"run_conformal_{model}.json.calib.partial.jsonl")
    if not os.path.exists(ckpt):
        return None
    recs = []
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def bad_v4(rec: dict) -> int:
    # Same folding-only definition as analyze_data_needed.py /
    # analyze_folding_only_definition.py: an actual correct -> incorrect
    # transition only; refusal is not silently counted as bad.
    return 1 if (rec["first_label"] == "correct" and rec["draft_label"] == "incorrect") else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--dir", type=str, default=DEFAULT_DIR)
    args = p.parse_args()
    alpha = args.alpha

    print(f"Target alpha = {alpha}, results dir = {args.dir}\n")
    header = f"{'model':<10}{'n_calib':<9}{'wilson_tau':<12}{'exact_tau':<11}{'exact_xi':<10}{'alpha_min':<10}verdict"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        recs = load_calib_records(args.dir, model)
        if recs is None:
            print(f"{model:<10}  MISSING checkpoint ({args.dir}/run_conformal_{model}.json.calib.partial.jsonl)")
            continue

        scores = [float(r["risk_score"]) for r in recs]
        bad = [bad_v4(r) for r in recs]

        wilson_tau = fit_global_threshold(scores, bad, alpha)
        exact_tau = fit_global_threshold_exact_crc(scores, bad, alpha)
        sel = certifiable_fraction(scores, bad, alpha)

        if wilson_tau < 0 and exact_tau >= 0:
            verdict = "RESCUED: bound-choice artifact, not a real ceiling"
        elif wilson_tau < 0 and exact_tau < 0:
            verdict = "STILL FAILS: real ceiling regardless of bound"
        elif wilson_tau >= 0 and exact_tau >= 0:
            verdict = "both certify"
        else:
            verdict = "exact fails where wilson succeeded (unexpected)"

        print(
            f"{model:<10}{len(recs):<9}{wilson_tau:<12.3f}{exact_tau:<11.3f}"
            f"{sel.xi:<10.3f}{sel.alpha_min:<10.3f}{verdict}"
        )


if __name__ == "__main__":
    main()
