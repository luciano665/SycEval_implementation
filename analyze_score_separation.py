"""
Does the risk score actually separate bad (regressive) calibration
instances from good ones, or is it just noise?

Two numbers per run, computed from the real stored calibration data
(<out>.calib.partial.jsonl -- same source as analyze_alpha_sensitivity.py,
no new LLM calls, no reruns):

  mean_bad - mean_good
      Average risk score of instances that turned out regressive, minus
      the average score of instances that didn't. Positive and large =
      bad instances score higher, as intended. Near zero = the score
      doesn't distinguish them at all.

  AUC (rank-based, Mann-Whitney U -- no sklearn/scipy dependency)
      Probability that a randomly chosen BAD instance scores higher than
      a randomly chosen GOOD instance.
        0.5  = no better than a coin flip (score carries no information)
        0.7+ = usable signal
        0.9+ = strong separation
      This is the standard way to quantify "does this score rank the two
      classes apart" independent of what threshold you'd pick.

Usage (from repo root, on the HPC, after the real suite has run):
    python3 analyze_score_separation.py
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


def rank_biserial_auc(scores, bad):
    """Mann-Whitney-U-based AUC: P(random bad score > random good score),
    with ties split 0.5/0.5. Pure Python, no numpy/scipy needed."""
    n_pos = sum(bad)
    n_neg = len(bad) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # undefined -- only one class present

    # rank all scores (average rank for ties), 1-indexed
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # average of positions i+1..j+1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[i] for i in range(len(scores)) if bad[i] == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def main():
    print(f"{'dataset':<12}{'model':<10}{'oracle':<11}{'n':<7}{'mean_bad':<10}"
          f"{'mean_good':<11}{'diff':<9}{'AUC':<8}interpretation")
    print("-" * 100)

    for dataset, model, oracle, out_path in RUNS:
        loaded = load_calib(out_path)
        if loaded is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}  MISSING: {out_path}.calib.partial.jsonl")
            continue
        scores, bad = loaded
        bad_scores = [s for s, b in zip(scores, bad) if b == 1]
        good_scores = [s for s, b in zip(scores, bad) if b == 0]
        mean_bad = sum(bad_scores) / len(bad_scores) if bad_scores else float("nan")
        mean_good = sum(good_scores) / len(good_scores) if good_scores else float("nan")
        diff = mean_bad - mean_good
        auc = rank_biserial_auc(scores, bad)
        auc_str = f"{auc:.3f}" if auc is not None else "n/a"

        if auc is None:
            interp = "only one class present"
        elif auc >= 0.85:
            interp = "strong separation"
        elif auc >= 0.65:
            interp = "moderate separation"
        elif auc >= 0.55:
            interp = "weak separation"
        else:
            interp = "~noise, no real signal"

        print(f"{dataset:<12}{model:<10}{oracle:<11}{len(scores):<7}{mean_bad:<10.3f}"
              f"{mean_good:<11.3f}{diff:<9.3f}{auc_str:<8}{interp}")


if __name__ == "__main__":
    main()
