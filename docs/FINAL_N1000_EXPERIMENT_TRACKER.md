# Final N=1000 Experiment Tracker

*Branch: `conformal_v10` (= `LM_corrections`). This is the paper's final,
reported experiment — a scaled-up version of `V9_EXPERIMENT_TRACKER.md`
/ `V10_EXPERIMENT_TRACKER.md`, run after both mentor-requested
diagnostics (`FINAL_DIAGNOSTICS_TRACKER.md`) were complete. Motivation:
N=300 already supported every finding in this project, but reads as
small for a paper by general research-paper norms, and a genuinely
larger, more precisely-bounded final dataset is worth having once the
core diagnostic story was locked in.*

---

## What's different from v9n / v10

Same protocol otherwise (deployable-only, leak-free rewrite,
folding-only labels, `threshold_method=exact_crc`, alpha=0.10, MedQuad):

| | v9n / v10 | This suite |
|---|---|---|
| N (total items) | 300 | **1000** |
| calib_frac | 0.5 (150 calib / 150 test) | **0.2 (200 calib / 800 test)** |
| baseline | reused from v9n | **re-run at N=1000** (12 jobs total, not 6) |

**Why 200/800, not a 50/50 or round-number split**: chosen via
`analyze_calib_size_projection.py` against the real v9 calibration data
before committing compute — calibration's achievable benefit (via the
exact-CRC bound's `(k+1)/(n+1)` finite-sample padding) saturates by
~200 items for the one model it was projected to affect (Gemma-1B: 200
vs 300 vs 400 calib items gave near-identical projected `alpha_min`).
Test-set precision keeps improving with more data, unlike calibration,
so the remaining items were allocated there.

**Pre-launch verification**: `smoke_test_final_n1000.slurm` (job
`142118`, N=20, `calib_frac=0.2` — never used before this suite,
previously always 0.5). COMPLETED, exit `0:0`, 51:27 elapsed. Confirmed
via `data_split.calib_question_hashes`: exactly 4 of 20 items (20%,
matching `calib_frac=0.2`) — split arithmetic verified correct before
committing multi-day compute to the real run.

## Suite launch status

**Status: ALL 12 COMPLETED**, exit `0:0` — submitted 2026-09-09,
confirmed complete 2026-09-11.

| Job ID | Model | Arm | Elapsed |
|---|---|---|---|
| 142119 | Llama-1B | baseline | 14:46:32 |
| 142120 | Llama-1B | conformal | 1d 3:30:58 |
| 142121 | Llama-3B | baseline | 20:22:40 |
| 142122 | Llama-3B | conformal | 1d 23:45:48 |
| 142123 | Gemma-1B | baseline | 21:40:30 |
| 142124 | Gemma-1B | conformal | 1d 15:54:29 |
| 142125 | Gemma-4B | baseline | 1d 4:26:44 |
| 142126 | Gemma-4B | conformal | 1d 23:42:15 |
| 142127 | Phi-1.5 | baseline | 1d 0:02:35 |
| 142128 | Phi-1.5 | conformal | 1d 7:38:18 |
| 142129 | Phi-2 | baseline | 1d 3:39:19 |
| 142130 | Phi-2 | conformal | 1d 10:52:50 |

Baseline jobs ran on `gpu_2day`; conformal jobs on `gpu_7day` with a
3-day budget (our slowest N=300 conformal job took 13:19:08 — scaled
~3.3x could approach 44+ hours, too close to `gpu_2day`'s 48h cap for
comfort). All finished comfortably within budget.

## Results: calibration verdicts (alpha=0.10, exact_crc)

Confirmed via each model's `thresholds_<model>.json`
(`max_items=1000, n_loaded=1000, n_calib_hashes=200` for all 6 — split
landed correctly):

| Model | tau_global | calibration_failed | alpha_min |
|---|---|---|---|
| Llama-1B | -1.0 | True | 0.120 |
| Llama-3B | -1.0 | True | 0.150 |
| Gemma-1B | -1.0 | True | 0.124 |
| Gemma-4B | -1.0 | True | 0.184 |
| Phi-1.5 | 1.0 | False | 0.023 |
| Phi-2 | 1.0 | False | 0.048 |

**Pass/fail at 10% is unchanged from N=300: still 2 of 6 (Phi-1.5,
Phi-2).**

## The projection didn't hold — real data diverged in both directions

`analyze_calib_size_projection.py`'s prediction (holding the N=300
observed rate constant, scaled up) turned out to be wrong once real new
data was collected — worth reporting as its own honest finding, not
just quietly using the real numbers:

| Model | N=300 (150 calib) | Projected (~200-300 calib) | **Real N=1000 (200 calib)** |
|---|---|---|---|
| Gemma-1B | 0.102 | ~0.100 (projected to pass) | **0.124 — worse, still fails** |
| Llama-1B | 0.115 | ~0.114 | **0.120 — slightly worse** |
| Llama-3B | 0.254 | ~0.253 (projected flat) | **0.150 — much better** |
| Gemma-4B | 0.260 | ~0.259 (projected flat) | **0.184 — much better** (matches N=300's *oracle*-assisted result almost exactly) |
| Phi-1.5 | 0.032 | — | 0.023 — better |
| Phi-2 | 0.053 | — | 0.048 — better |

The projection's assumption — that more calibration data would show the
*same* observed rate with tighter statistical padding — didn't hold.
Real data revealed genuinely different underlying rates for Llama-3B
and Gemma-4B (much closer to passing) and for Gemma-1B (further from
passing), not just a tightened bound on an unchanged rate. This is the
real, trustworthy result; the projection was a cheap pre-check, not a
substitute for actually collecting the data — treat it as validated
(and shown to have real limits) rather than confirmed.

## Results locations

`results/final_n1000_medquad/run_baseline_<model>.json`,
`run_conformal_<model>.json`, `thresholds_<model>.json`.

## Next steps

1. Headline baseline-vs-conformal analysis (paired to the 800-item test
   split, same pattern as `analyze_v9_headline.py` but pointed at this
   directory and this calib_frac) — not yet run.
2. This is the dataset to report in the paper's main results table —
   supersedes the N=300 v9n numbers as the primary reported result,
   though v9n's numbers remain valid as the smaller-scale, independently
   cross-validated version (Wilson vs. exact-CRC agreement, oracle test)
   that the diagnostic story was actually built on.
