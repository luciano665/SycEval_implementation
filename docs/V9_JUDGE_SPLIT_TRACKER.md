# Independent-Judge (Mistral) Suite Tracker

*Branch: `conformal_v9`. Companion to `V9_EXPERIMENT_TRACKER.md` (the
original v9 suite, Qwen doing every referee role). This suite tests
whether separating the judge from the rebuttal-writer/risk-scorer changes
the headline numbers — motivated by the self-preference-bias concern
(one model grading, persuading, and risk-scoring all being the same
model/family).*

---

## What's different from v9n

Everything else identical (deployable-only, leak-free rewrite,
folding-only labels, global thresholds, alpha=0.10, N=300 MedQuad,
calib_frac=0.5) — only the referee split changes:

| Role | v9n | v9j (this suite) |
|---|---|---|
| judge_model (`judge_local`, `judge_claim_support`) | Qwen2.5-7B-Instruct | **Mistral-7B-Instruct-v0.3** |
| rebuttal_model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct (unchanged) |
| risk_scorer_model | Qwen2.5-7B-Instruct (via default fallback) | Qwen2.5-7B-Instruct (**explicit** — the fallback would otherwise silently inherit judge_model=Mistral) |

## Pre-launch verification

`smoke_test_referee_split_gemma_1b.slurm` (job `135218`, N=20):
**COMPLETED**, exit `0:0`, 1:37:32 elapsed (longer than the equivalent
v9n smoke test's 43:53 — not investigated further, no errors/retries
found, likely just Mistral being slower per-call). Confirmed via the
output config block: `judge_model: Mistral-7B-Instruct-v0.3`,
`rebuttal_model` and `risk_scorer_model` both `Qwen2.5-7B-Instruct` — no
silent fallback. `calibration_failed: true` at N=20 is expected (same as
the v9n smoke test showed) — not a signal at this tiny N.

## Suite launch status

**Status: ALL 12 SUBMITTED** — submitted 2026-08-24.

| Job ID | Model | Arm |
|---|---|---|
| 135674 | Llama-1B | baseline |
| 135675 | Llama-1B | conformal |
| 135676 | Llama-3B | baseline |
| 135677 | Llama-3B | conformal |
| 135678 | Gemma-1B | baseline |
| 135679 | Gemma-1B | conformal |
| 135680 | Gemma-4B | baseline |
| 135681 | Gemma-4B | conformal |
| 135682 | Phi-1.5 | baseline |
| 135683 | Phi-1.5 | conformal |
| 135684 | Phi-2 | baseline |
| 135685 | Phi-2 | conformal |

Check status: `sacct -j 135674-135685 --format=JobID,JobName,State,ExitCode,Elapsed`

## Results locations

`results/v9_judge_split_medquad/run_baseline_<model>.json`,
`run_conformal_<model>.json`, `thresholds_<model>.json`.

## Next steps once complete

1. Check every thresholds file: real `tau_global` or still `-1.0`? Compare
   per-model against the v9n (Qwen-judge) results already on record.
2. Re-run `analyze_v9_headline.py` pointed at
   `results/v9_judge_split_medquad` (swap the `DIR` constant) to get
   paired baseline-vs-conformal rates under the new judge.
3. **The real question**: does switching judge from Qwen to Mistral
   change which models calibrate, or change the regressive/progressive
   rates meaningfully? If judge choice barely moves the numbers, that's
   evidence self-preference bias isn't a major factor in this pipeline.
   If it moves them a lot, that's a real finding worth flagging in the
   writeup — and possibly grounds to also separate risk-scoring onto a
   third model (deferred for now, see conversation reasoning: risk-scorer
   bias is calibration-robust in a way judge bias isn't, so it was lower
   priority to fix first).
