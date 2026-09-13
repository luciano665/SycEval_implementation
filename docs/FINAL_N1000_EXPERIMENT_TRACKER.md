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

## Results: rates (MedQuad, N=1000)

Two measurements, and they disagree — the paired one is the trustworthy
one.

**Paired before/after rewrite** (`analyze_rewrite_effect_v10.py`, same
draft compared to itself, `rewrite_rate=1.000` for all four failing
models since `tau=-1` means rewrite-everything):

| Model | regr_before | regr_after | delta | verdict |
|---|---|---|---|---|
| llama_1b | 0.134 | 0.178 | **+0.044** | rewrite hurts |
| llama_3b | 0.285 | 0.338 | **+0.052** | rewrite hurts |
| gemma_1b | 0.129 | 0.179 | **+0.050** | rewrite hurts |
| gemma_4b | 0.281 | 0.296 | **+0.015** | roughly neutral |
| phi_1.5 / phi_2 | n/a | n/a | n/a | 0 rewrites triggered |

**Rewrite never helps any model, and now hurts 3 of 4** (at N=300 it hurt
2 of 4 — Llama-1B moved from neutral to harmful).

**Headline baseline-vs-conformal** (`analyze_final_n1000_headline.py`,
two separate runs) gives different, partly contradictory deltas:
llama_1b +0.030, llama_3b −0.000, gemma_1b −0.021, gemma_4b **−0.102**,
phi_1.5 −0.015, phi_2 −0.034.

**Do not report Gemma-4B's −0.102 as a rewrite effect.** The paired
measurement puts the real effect at +0.015, essentially neutral. This
is the same artifact already documented at N=300 (see
`FINAL_DIAGNOSTICS_TRACKER.md`: the old "Gemma-4B helps a lot" claim
traced to the C1 oracle leak), reappearing because the headline method
compares two independently-sampled runs rather than the same draft
before and after.

**Unexplained, worth checking before the writeup**: the conformal arm's
*pre-rewrite* drafts fold less than the separate baseline run —
Gemma-4B 0.281 vs 0.398, Gemma-1B 0.129 vs 0.200. Something before the
rewrite step accounts for the headline "improvement." Leading
hypothesis: the claim-level filter, which runs on every draft regardless
of the rewrite decision and is operating on the uncalibrated 0.3
fallback (`tau_claim_fallback=True` for all models). Not verified.

---

## Second dataset: HealthSearchQA (N=1000)

Same protocol, only `--domain` changes — deployable-only, leak-free
rewrite, folding-only labels, alpha=0.10, `threshold_method=exact_crc`,
`calib_frac=0.2` (200 calib / 800 test). Holding every other knob fixed
is deliberate: any difference between the two datasets is then
attributable to the dataset rather than a config change.

The v6-era HealthSearchQA results are **not** reusable — `review_findings.md`
C1 (oracle truth leaking into the test-time intervention) applies to
them, and that leak is what produced the since-retracted "7.4%
reduction" figure.

**Pre-launch verification**: `smoke_test_final_hs.slurm` (job `143068`,
N=20). COMPLETED, exit `0:0`, 44:18. Confirmed 4 calib of 20 loaded
(`calib_frac=0.2`), 128 records (16 test items × 8 instances), real
HealthSearchQA questions, and coherent answer text — the field
difference from MedQuad (`Free_form_answer` vs `Answer`) flows through
the loader, judge, and claim decomposition correctly.

**Status: 12 jobs submitted 2026-09-13**, results pending.

| Job ID | Model | Arm |
|---|---|---|
| 144670 | Llama-1B | baseline |
| 144671 | Llama-1B | conformal |
| 144672 | Llama-3B | baseline |
| 144673 | Llama-3B | conformal |
| 144674 | Gemma-1B | baseline |
| 144675 | Gemma-1B | conformal |
| 144676 | Gemma-4B | baseline |
| 144677 | Gemma-4B | conformal |
| 144678 | Phi-1.5 | baseline |
| 144679 | Phi-1.5 | conformal |
| 144680 | Phi-2 | baseline |
| 144681 | Phi-2 | conformal |

## Results locations

MedQuad: `results/final_n1000_medquad/`
HealthSearchQA: `results/final_n1000_healthsearch/`
Both contain `run_baseline_<model>.json`, `run_conformal_<model>.json`,
`thresholds_<model>.json`.

## Next steps

1. When the HealthSearchQA suite finishes: calibration verdicts +
   `analyze_final_n1000_headline.py` / `analyze_rewrite_effect_v10.py`
   pointed at `results/final_n1000_healthsearch`.
2. Check the pre-rewrite gap noted above (conformal drafts folding less
   than baseline) before any headline number goes in the paper.
3. The oracle diagnostic ran at N=300 while these suites are N=1000. If
   both appear in the paper, either note the scale difference or rerun
   the oracle at N=1000 — calibration-only for 4 models, so relatively
   cheap.
4. These two suites are the paper's primary reported results; the N=300
   v9n numbers remain valid as the smaller-scale, independently
   cross-validated version (Wilson vs. exact-CRC agreement, oracle test)
   that the diagnostic story was built on.
