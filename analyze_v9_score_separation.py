"""
Does the risk scorer actually separate bad (regressive) calibration
instances from good ones in the REAL v9 suite -- per model?

Motivation: v9 calibration landed on one of two trivial extremes for
every model -- tau_global=-1.0 (reject/rewrite everything: Llama-1B/3B,
Gemma-1B/4B) or tau_global=1.0 (accept/rewrite nothing: Phi-1.5, Phi-2,
confirmed 0/1200 rewrite_triggered for Phi-1.5). Neither shows the
framework doing genuine selective calibration. Before concluding the
scorer needs to be rebuilt or group thresholds are the fix, check
whether the scorer has ANY real discriminative power per model -- this
reads calibration data already collected on disk, no new LLM calls.

Two numbers per model (same method as the original analyze_score_separation.py,
just pointed at v9 data):

  mean_bad - mean_good
      Average risk score of regressive calibration instances minus
      average score of non-regressive ones. Near zero = score carries no
      information about which instances actually turned out bad.

  AUC (rank-based, Mann-Whitney U)
      P(a random BAD instance scores higher than a random GOOD instance).
        0.5  = no better than a coin flip
        0.7+ = usable signal
        0.9+ = strong separation

Usage (from repo root, on the HPC, after the v9 suite has run):
    python3 analyze_v9_score_separation.py
"""
from __future__ import annotations
import json
import os

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
DIR = "results/v9_night_medquad"


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
    print(f"{'model':<10}{'n':<8}{'n_bad':<8}{'bad_rate':<10}{'mean_bad':<10}{'mean_good':<10}{'gap':<10}{'AUC':<8}")
    print("-" * 74)
    for model in MODELS:
        loaded = load_calib(model)
        if loaded is None:
            print(f"{model:<10}  MISSING calibration checkpoint")
            continue
        scores, bad = loaded
        n = len(scores)
        n_bad = sum(bad)
        if n == 0:
            print(f"{model:<10}  EMPTY")
            continue
        bad_rate = n_bad / n
        mean_bad = sum(s for s, b in zip(scores, bad) if b == 1) / max(1, n_bad)
        mean_good = sum(s for s, b in zip(scores, bad) if b == 0) / max(1, n - n_bad)
        gap = mean_bad - mean_good
        auc = rank_biserial_auc(scores, bad)
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"
        print(f"{model:<10}{n:<8}{n_bad:<8}{bad_rate:<10.3f}{mean_bad:<10.3f}{mean_good:<10.3f}{gap:<+10.3f}{auc_str:<8}")


if __name__ == "__main__":
    main()
