"""
For the models where calibration fails: is more calibration data even
capable of fixing it, and if so, roughly how much?

Key distinction this script draws out, per candidate threshold tau:
  - true bad rate at tau (p_hat = k/n) ALREADY >= alpha
      -> more data can NEVER help at this tau. The Wilson upper bound
         converges to the true rate as n grows; if the true rate itself
         exceeds the target, no amount of additional calibration data
         fixes it. This is a genuine behavioral ceiling, not a data
         shortage.
  - true bad rate at tau IS < alpha, but the current Wilson upper bound
    (which accounts for statistical uncertainty at the current small n)
    is still > alpha
      -> this IS just a data shortage. More calibration data (same
         underlying distribution) would tighten the confidence interval
         around the already-good true rate and eventually succeed.

For each model/config that currently fails at the target alpha, this
finds the most permissive candidate threshold whose TRUE rate is still
under alpha, then estimates via binary search how large the calibration
set would need to be (scaling n and k proportionally, i.e. assuming more
of the same distribution, not a different one) for the Wilson bound to
actually drop below alpha.

Uses only already-stored calibration checkpoint data (bad_v4 definition,
the corrected one) -- no new LLM calls, no reruns.

Usage (from repo root, on the HPC, after the real suite has run):
    python3 analyze_data_needed.py --alpha 0.15
"""
from __future__ import annotations
import argparse
import json
import os

from conformal_v2.conformal_thresholds import wilson_upper_bound

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


def best_candidate_under_true_rate(scores, bad, alpha):
    """Among all candidate thresholds, find the most permissive one (max n)
    whose TRUE rate (not the Wilson-inflated bound) is still under alpha.
    Returns (tau, n, k, p_hat) or None if no candidate has a true rate
    under alpha even at the strictest possible acceptance set."""
    candidates = sorted(set(scores))
    best = None  # (n, tau, k, p_hat)
    for tau in candidates:
        idx = [i for i, s in enumerate(scores) if s <= tau]
        n = len(idx)
        if n == 0:
            continue
        k = sum(bad[i] for i in idx)
        p_hat = k / n
        if p_hat < alpha:
            if best is None or n > best[0]:
                best = (n, tau, k, p_hat)
    return best


def data_multiplier_needed(k, n, alpha, z=1.96, max_multiplier=200):
    """Binary search: smallest multiplier m such that scaling (k, n) by m
    (same p_hat, more data) brings the Wilson upper bound under alpha."""
    if wilson_upper_bound(k, n, z) <= alpha:
        return 1.0
    lo, hi = 1.0, float(max_multiplier)
    if wilson_upper_bound(round(k * hi), round(n * hi), z) > alpha:
        return None  # even max_multiplier isn't enough to report a finite estimate
    for _ in range(40):
        mid = (lo + hi) / 2
        ub = wilson_upper_bound(round(k * mid), round(n * mid), z)
        if ub <= alpha:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.15)
    args = p.parse_args()
    alpha = args.alpha

    print(f"Target alpha = {alpha}\n")
    print(f"{'dataset':<12}{'model':<10}{'oracle':<11}{'verdict':<45}{'detail'}")
    print("-" * 130)

    for dataset, model, oracle, out_path in RUNS:
        recs = load_calib_full(out_path)
        if recs is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}  MISSING checkpoint")
            continue

        scores = [float(r["risk_score"]) for r in recs]
        bad_v4 = [1 if (r["first_label"] == "correct" and r["draft_label"] == "incorrect") else 0 for r in recs]

        best = best_candidate_under_true_rate(scores, bad_v4, alpha)
        if best is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}"
                  f"{'NO AMOUNT OF DATA CAN FIX THIS':<45}"
                  f"true rate exceeds {alpha} at every possible threshold")
            continue

        n, tau, k, p_hat = best
        mult = data_multiplier_needed(k, n, alpha)
        current_n_calib_items = len(recs) / 8  # 8 rebuttal steps per item
        if mult is None:
            print(f"{dataset:<12}{model:<10}{oracle:<11}"
                  f"{'DATA SHORTAGE, but needs >200x more':<45}"
                  f"best true rate {p_hat:.3f} at n={n}, still not practically reachable")
        elif mult <= 1.01:
            print(f"{dataset:<12}{model:<10}{oracle:<11}"
                  f"{'ALREADY SUCCEEDS':<45}"
                  f"(should not appear if this model failed at this alpha)")
        else:
            new_n_calib_items = current_n_calib_items * mult
            print(f"{dataset:<12}{model:<10}{oracle:<11}"
                  f"{'DATA SHORTAGE — genuinely fixable':<45}"
                  f"needs ~{mult:.1f}x more calibration data "
                  f"(~{current_n_calib_items:.0f} -> ~{new_n_calib_items:.0f} calibration items)")


if __name__ == "__main__":
    main()
