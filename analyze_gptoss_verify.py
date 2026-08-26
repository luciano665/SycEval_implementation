"""
Does gpt-oss-20b as risk_scorer_model close the AUC gap that v3's prompt
fix (same Qwen-7B model, better prompt) couldn't? Reads the calibration
checkpoints from verify_gptoss_scorer_*.slurm (N=60 each, llama_1b and
gemma_1b -- the same two borderline cases used for the v3 verification,
for a direct three-way comparison).

Baseline numbers already on record:
  llama_1b: AUC(old)=0.302 (inverted), AUC(v3)=0.470, bad_rate=0.133
  gemma_1b: AUC(old)=0.508 (no signal), AUC(v3)=0.575, bad_rate=0.113

Usage (from repo root, on the HPC, after both verify_gptoss_scorer_*.slurm
jobs complete):
    python3 analyze_gptoss_verify.py
"""
from __future__ import annotations
import json
import os

MODELS = ["llama_1b", "gemma_1b"]
DIR = "results/verify_gptoss_scorer"

PRIOR_RESULTS = {
    "llama_1b": dict(auc_old=0.302, auc_v3=0.470, bad_rate=0.133),
    "gemma_1b": dict(auc_old=0.508, auc_v3=0.575, bad_rate=0.113),
}


def rank_biserial_auc(scores, bad):
    n_pos = sum(bad)
    n_neg = len(bad) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[i] for i in range(len(scores)) if bad[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def load_calib(model):
    ckpt = f"{DIR}/run_conformal_{model}.json.calib.partial.jsonl"
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
    print(f"{'model':<10}{'n':<6}{'bad_rate':<10}{'AUC(old)':<10}{'AUC(v3)':<10}{'AUC(gptoss)':<12}{'unique':<8}")
    print("-" * 66)
    for model in MODELS:
        loaded = load_calib(model)
        if loaded is None:
            print(f"{model:<10}  MISSING -- has verify_gptoss_scorer_{model}.slurm completed?")
            continue
        scores, bad = loaded
        n = len(scores)
        if n == 0:
            print(f"{model:<10}  EMPTY")
            continue
        bad_rate = sum(bad) / n
        auc = rank_biserial_auc(scores, bad)
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        prior = PRIOR_RESULTS[model]
        print(f"{model:<10}{n:<6}{bad_rate:<10.3f}{prior['auc_old']:<10.3f}{prior['auc_v3']:<10.3f}{auc_str:<12}{len(set(scores)):<8}")

        thresh_path = f"{DIR}/thresholds_{model}.json"
        if os.path.exists(thresh_path):
            t = json.load(open(thresh_path))
            print(f"{'':<10}tau_global={t.get('tau_global')}  calibration_failed={t.get('calibration_failed')}")


if __name__ == "__main__":
    main()
