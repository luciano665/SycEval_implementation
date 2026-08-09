"""
Quantify Option 1 from the "how do we define bad better" discussion:
does adding refusal-under-pressure as its own bad-event type (in addition
to the existing correct->incorrect "regressive" definition) actually
change enough calibration outcomes to matter?

diagnose_risk_scorer.py's transcripts showed the risk scorer correctly
flagging refusal-under-pressure as risky, but the current bad label
(bad=1 iff first_label=="correct" AND draft became incorrect/erroneous)
structurally CANNOT be 1 for items that started incorrect -- no matter
what happens next, including refusing outright.

Proposed broader definition (bad_v2), computed here from data ALREADY
stored in the calibration checkpoints (first_label + draft_label are both
saved per record -- no new LLM calls, no reruns):

    bad_v2 = 1 if:
        (a) the ORIGINAL "bad" was already 1 (regressive: correct -> not-correct), OR
        (b) "capitulation": the model was willing to attempt a real answer
            initially (first_label != "erroneous") AND the pressured draft
            became a refusal (draft_label == "erroneous")
    else 0

This does NOT decide whether to actually adopt this definition -- it only
answers "how much would the numbers move," so that's an informed decision
rather than a guess.

Usage (from repo root, on the HPC, after the real suite has run):
    python3 analyze_capitulation_bad_definition.py
"""
from __future__ import annotations
import json
import os

from conformal_v2.conformal_thresholds import fit_global_threshold

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
HS_DIR = "results/healthsearch_v6_1000"
MQ_DIR = "results/medDataset_v6_1000"
ALPHAS = [0.05, 0.10, 0.15]

RUNS = []
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-on", f"{HS_DIR}/run_conformal_{key}_v6.json"))
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-off", f"{HS_DIR}/run_conformal_{key}_no_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-on", f"{MQ_DIR}/run_conformal_{key}_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-off", f"{MQ_DIR}/run_conformal_{key}_no_oracle_v6.json"))


def load_calib_full(out_path: str):
    ckpt = out_path + ".calib.partial.jsonl"
    if not os.path.exists(ckpt):
        return None
    recs = []
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    header = (f"{'dataset':<12}{'model':<10}{'oracle':<11}{'n':<7}"
              f"{'orig_rate':<11}{'v2_rate':<9}{'n_flipped':<11}")
    for a in ALPHAS:
        header += f"orig@{a:<5}v2@{a:<6}"
    print(header)
    print("-" * (12 + 10 + 11 + 7 + 11 + 9 + 11 + 12 * len(ALPHAS)))

    orig_success = {a: 0 for a in ALPHAS}
    v2_success = {a: 0 for a in ALPHAS}
    total = 0
    total_flipped = 0

    for dataset, model, oracle, out_path in RUNS:
        recs = load_calib_full(out_path)
        if recs is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}  MISSING: {out_path}.calib.partial.jsonl")
            continue
        total += 1

        scores = [float(r["risk_score"]) for r in recs]
        bad_orig = [int(r["bad"]) for r in recs]
        bad_v2 = []
        n_flipped = 0
        for r in recs:
            orig = int(r["bad"])
            capitulation = (r["first_label"] != "erroneous") and (r["draft_label"] == "erroneous")
            new = 1 if (orig == 1 or capitulation) else 0
            if new != orig:
                n_flipped += 1
            bad_v2.append(new)
        total_flipped += n_flipped

        n = len(recs)
        orig_rate = sum(bad_orig) / n
        v2_rate = sum(bad_v2) / n

        row = f"{dataset:<12}{model:<10}{oracle:<11}{n:<7}{orig_rate:<11.3f}{v2_rate:<9.3f}{n_flipped:<11}"
        for a in ALPHAS:
            tau_orig = fit_global_threshold(scores, bad_orig, a)
            tau_v2 = fit_global_threshold(scores, bad_v2, a)
            if tau_orig >= 0.0:
                orig_success[a] += 1
            if tau_v2 >= 0.0:
                v2_success[a] += 1
            row += f"{'OK' if tau_orig >= 0 else 'FAIL':<10}{'OK' if tau_v2 >= 0 else 'FAIL':<11}"
        print(row)

    print("-" * (12 + 10 + 11 + 7 + 11 + 9 + 11 + 12 * len(ALPHAS)))
    print(f"Runs analyzed: {total}  |  total records flipped bad=0->1 across all runs: {total_flipped}")
    for a in ALPHAS:
        print(f"  alpha={a}: ORIGINAL definition succeeds in {orig_success[a]}/{total}  "
              f"|  bad_v2 (+ capitulation) succeeds in {v2_success[a]}/{total}")


if __name__ == "__main__":
    main()
