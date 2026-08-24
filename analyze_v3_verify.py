"""
Did deploying the v3 risk-scorer prompt to production actually improve
AUC? Reads the calibration checkpoints from the verify_v3_scorer_*.slurm
jobs (N=60 each, llama_1b and gemma_1b -- the two borderline cases from
v9n where a working scorer might rescue calibration).

Compare against the v9n (old-scorer) baseline already on record:
  llama_1b: AUC=0.302 (inverted), bad_rate=0.114
  gemma_1b: AUC=0.508 (no signal), bad_rate=0.116

Usage (from repo root, on the HPC, after both verify_v3_scorer_*.slurm
jobs complete):
    python3 analyze_v3_verify.py
"""
from __future__ import annotations
import json
import os

MODELS = ["llama_1b", "gemma_1b"]
DIR = "results/verify_v3_scorer"

OLD_SCORER_BASELINE = {
    "llama_1b": dict(auc=0.302, bad_rate=0.114),
    "gemma_1b": dict(auc=0.508, bad_rate=0.116),
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
    print(f"{'model':<10}{'n':<8}{'bad_rate':<10}{'AUC(v3)':<10}{'AUC(old)':<10}{'delta':<8}")
    print("-" * 56)
    for model in MODELS:
        loaded = load_calib(model)
        if loaded is None:
            print(f"{model:<10}  MISSING -- has verify_v3_scorer_{model}.slurm completed?")
            continue
        scores, bad = loaded
        n = len(scores)
        if n == 0:
            print(f"{model:<10}  EMPTY")
            continue
        bad_rate = sum(bad) / n
        auc = rank_biserial_auc(scores, bad)
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        old = OLD_SCORER_BASELINE[model]
        delta = f"{auc - old['auc']:+.3f}" if auc is not None else "n/a"
        print(f"{model:<10}{n:<8}{bad_rate:<10.3f}{auc_str:<10}{old['auc']:<10.3f}{delta:<8}")

        thresh_path = f"{DIR}/thresholds_{model}.json"
        if os.path.exists(thresh_path):
            t = json.load(open(thresh_path))
            print(f"{'':<10}tau_global={t.get('tau_global')}  calibration_failed={t.get('calibration_failed')}")


if __name__ == "__main__":
    main()
