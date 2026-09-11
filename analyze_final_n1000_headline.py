"""
Headline numbers for the final N=1000 suite: baseline vs conformal
sycophancy rates, per model, properly paired.

Same script as analyze_v9_headline.py, pointed at the N=1000 results
directory. Baseline runs on all 1000 items; conformal tests on the
800-item test split (200 calibration items held out, calib_frac=0.2).
Comparing baseline's full 1000 against conformal's 800 would be the same
unpaired-comparison bug fixed for the old V6 suite (C13) -- this
restricts baseline to the same test-split items conformal was actually
evaluated on, using the calibration item hashes stored in each model's
thresholds file.

Usage (from repo root, on the HPC, after the final N=1000 suite has run):
    python3 analyze_final_n1000_headline.py
"""
from __future__ import annotations
import json
import os

from conformal_v2.run_conformal_v2 import _question_hash

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
DIR = "results/final_n1000_medquad"


def load_records(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d.get("records") or d.get("individual_records") or d.get("test_instances", [])


def rates(records):
    n = len(records)
    if n == 0:
        return dict(N=0, regressive=0.0, progressive=0.0, none=0.0)
    regr = sum(1 for r in records if r["sycophancy"] == "regressive") / n
    prog = sum(1 for r in records if r["sycophancy"] == "progressive") / n
    none = sum(1 for r in records if r["sycophancy"] == "none") / n
    return dict(N=n, regressive=regr, progressive=prog, none=none)


def main():
    print(f"{'model':<10}{'arm':<12}{'N':<6}{'regressive':<12}{'progressive':<12}{'none':<8}")
    print("-" * 60)

    for model in MODELS:
        thresh_path = f"{DIR}/thresholds_{model}.json"
        baseline_path = f"{DIR}/run_baseline_{model}.json"
        conformal_path = f"{DIR}/run_conformal_{model}.json"

        if not os.path.exists(thresh_path):
            print(f"{model:<10}  MISSING: {thresh_path}")
            continue

        thresh = json.load(open(thresh_path))
        calib_hashes = set(thresh.get("data_split", {}).get("calib_question_hashes", []))

        baseline_recs = load_records(baseline_path)
        conformal_recs = load_records(conformal_path)

        if baseline_recs is None or conformal_recs is None:
            print(f"{model:<10}  MISSING output file(s)")
            continue

        # pair: restrict baseline to the same test-split questions conformal used
        baseline_paired = [r for r in baseline_recs if _question_hash(r["question"]) not in calib_hashes]

        b = rates(baseline_paired)
        c = rates(conformal_recs)

        print(f"{model:<10}{'baseline':<12}{b['N']:<6}{b['regressive']:<12.3f}{b['progressive']:<12.3f}{b['none']:<8.3f}")
        print(f"{model:<10}{'conformal':<12}{c['N']:<6}{c['regressive']:<12.3f}{c['progressive']:<12.3f}{c['none']:<8.3f}")
        delta = c['regressive'] - b['regressive']
        print(f"{'':<10}{'delta regr':<12}{'':<6}{delta:<+12.3f}")
        print()


if __name__ == "__main__":
    main()
