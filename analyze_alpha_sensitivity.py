"""
Post-hoc "what if we used a different alpha" analysis, using the REAL
fit_global_threshold function against the REAL stored calibration data --
not a guess, not a re-simulation with different code. Answers: "if we'd
set --alpha to 0.10 or 0.15 instead of 0.05, would calibration have
actually succeeded for each of the 24 conformal runs?"

No new LLM calls, no reruns -- this only reads the calibration checkpoint
JSONL files that were already written during the real run
(<out>.calib.partial.jsonl, one line per calibration rebuttal-step with a
"risk_score" and "bad" field -- see calibration_collect() in
conformal_v2/run_conformal_v2.py).

Usage (from repo root, on the HPC, after the real suite has run):
    python3 analyze_alpha_sensitivity.py
"""
from __future__ import annotations
import json
import os

from conformal_v2.conformal_thresholds import fit_global_threshold

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
HS_DIR = "results/healthsearch_v6_1000"
MQ_DIR = "results/medDataset_v6_1000"
ALPHAS = [0.05, 0.10, 0.15, 0.20]

RUNS = []
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-on", f"{HS_DIR}/run_conformal_{key}_v6.json"))
for key in MODELS:
    RUNS.append(("healthsearch", key, "oracle-off", f"{HS_DIR}/run_conformal_{key}_no_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-on", f"{MQ_DIR}/run_conformal_{key}_oracle_v6.json"))
for key in MODELS:
    RUNS.append(("medquad", key, "oracle-off", f"{MQ_DIR}/run_conformal_{key}_no_oracle_v6.json"))


def load_calib(out_path: str):
    ckpt = out_path + ".calib.partial.jsonl"
    if not os.path.exists(ckpt):
        return None
    scores, bad = [], []
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scores.append(float(r["risk_score"]))
            bad.append(int(r["bad"]))
    return scores, bad


def main():
    header = f"{'dataset':<12}{'model':<10}{'oracle':<11}{'n_calib':<9}{'raw_regr_rate':<15}"
    for a in ALPHAS:
        header += f"tau@{a:<6}"
    print(header)
    print("-" * (12 + 10 + 11 + 9 + 15 + 11 * len(ALPHAS)))

    succeeded_at = {a: 0 for a in ALPHAS}
    total = 0

    for dataset, model, oracle, out_path in RUNS:
        loaded = load_calib(out_path)
        if loaded is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}  MISSING CHECKPOINT: {out_path}.calib.partial.jsonl")
            continue
        scores, bad = loaded
        total += 1
        n = len(scores)
        raw_rate = sum(bad) / n if n else float("nan")

        row = f"{dataset:<12}{model:<10}{oracle:<11}{n:<9}{raw_rate:<15.3f}"
        for a in ALPHAS:
            tau = fit_global_threshold(scores, bad, a)
            if tau >= 0.0:
                succeeded_at[a] += 1
                row += f"{tau:<11.3f}"
            else:
                row += f"{'FAIL':<11}"
        print(row)

    print("-" * (12 + 10 + 11 + 9 + 15 + 11 * len(ALPHAS)))
    print(f"Runs analyzed: {total}")
    for a in ALPHAS:
        print(f"  alpha={a}: calibration succeeds in {succeeded_at[a]}/{total} runs")


if __name__ == "__main__":
    main()
