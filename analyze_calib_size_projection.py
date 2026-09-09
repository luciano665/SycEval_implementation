"""
Free, no-new-compute check: would a BIGGER calibration set alone (same
scorer, same observed bad-rate, just more calibration items) push any of
the still-failing models under the alpha=0.10 target?

Motivation: the exact-CRC bound is (k+1)/(n+1). At small n, the "+1"
adds real conservative padding -- e.g. 0 bad out of 15 accepted still
bounds at 1/16 = 6.25%, not 0%. A bigger calibration set shrinks that
padding's relative effect, which could rescue a borderline model WITHOUT
needing a better scorer at all. This is a different, complementary
hypothesis to the oracle test (which asked "is the scorer's blindness
the problem") -- this asks "is calibration-set SIZE itself part of the
problem, independent of the scorer."

Method: for each candidate tau already in the real v9 calibration data,
take the observed (k, n) -- bad count and accepted count -- and project
what exact_crc_bound(k, n) would look like if we had `scale`x as much
calibration data at the SAME observed bad-rate (k*scale, n*scale, both
rounded). Report the best (minimum) projected bound across all
candidates -- i.e. a projected alpha_min -- at scale=1 (matches the real
alpha_min exactly, sanity check), 2x (300 calib items), and 3x (450).

This is a PROJECTION, not a new experiment: it assumes the same observed
rate holds at a larger N, which real new data might not confirm (a
larger sample could reveal a different true rate, tighter or looser).
It only tells us whether the idea is worth spending real compute on.

Usage (from repo root, on the HPC, after the v9 suite has run):
    python3 analyze_calib_size_projection.py
"""
from __future__ import annotations
import json
import os

from conformal_v2.conformal_thresholds import exact_crc_bound

MODELS = ["llama_1b", "llama_3b", "gemma_1b", "gemma_4b", "phi_1.5", "phi_2"]
DIR = "results/v9_night_medquad"
SCALES = [1, 2, 3]
ALPHA = 0.10


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


def projected_alpha_min(scores, bad, scale):
    """Best (minimum) exact_crc_bound across all candidate tau, with
    (k, n) scaled by `scale` at the same observed rate."""
    candidates = sorted(set(scores))
    best = 1.0
    for tau in candidates:
        accepted_idx = [i for i, s in enumerate(scores) if s <= tau]
        n = len(accepted_idx)
        if n == 0:
            continue
        k = sum(bad[i] for i in accepted_idx)
        n_scaled = round(n * scale)
        k_scaled = round(k * scale)
        bound = exact_crc_bound(k_scaled, n_scaled)
        if bound < best:
            best = bound
    return best


def main():
    header = f"{'model':<10}{'n_calib':<10}"
    for s in SCALES:
        header += f"alpha_min@{s}x".ljust(16)
    header += "would_pass_at_0.10?"
    print(header)
    print("-" * (10 + 10 + 16 * len(SCALES) + 22))

    for model in MODELS:
        loaded = load_calib(model)
        if loaded is None:
            print(f"{model:<10}  MISSING checkpoint")
            continue
        scores, bad = loaded
        n = len(scores)
        row = f"{model:<10}{n:<10}"
        projections = []
        for s in SCALES:
            am = projected_alpha_min(scores, bad, s)
            projections.append(am)
            row += f"{am:.4f}".ljust(16)
        crosses = any(p <= ALPHA for p in projections) and not (projections[0] <= ALPHA)
        note = "YES, at bigger N" if crosses else ("already passes" if projections[0] <= ALPHA else "no, stays failing")
        row += note
        print(row)


if __name__ == "__main__":
    main()
