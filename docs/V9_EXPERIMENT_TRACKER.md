# conformal_v9 Overnight Suite Tracker

*Branch: `conformal_v9`. Companion to `docs/EXPERIMENT_TRACKER.md` (the
older V6 suite, 2x2 oracle design, now superseded — oracle option removed
entirely in v9). See commits `a402d16` (deployable-only + leak-free rewrite)
and `a10c88e` (folding-only labels + this suite) for what changed and why.*

---

## What's different from V6

- **Deployable-only**: no oracle-truth option at all (removed from the
  whole pipeline, not just defaulted off).
- **Leak-free rewrite**: `anti_sycophancy_rewrite` now only sees
  `question` + the purified draft — never the rebuttal or initial answer.
- **Folding-only labels**: `classify_sychophancy` no longer counts refusal
  (`erroneous`) as sycophancy in either direction — only a genuine
  correct→incorrect flip counts as regressive.
- **Global thresholds only** — group thresholds dropped (fragment
  calibration at this N).
- **α = 0.10** (not 0.05) — the loosened target discussed as more
  defensible given V6's near-total calibration failure at 5%.
- **N = 300, MedQuad only** (not N=1000, not HealthSearchQA) — smaller,
  faster suite: `calib_frac=0.5` → 150 calibration / 150 test items.
- **12 jobs total** (6 baseline + 6 conformal), not 36 — no dataset ×
  oracle matrix this time, just one config per model family.

## Pre-launch verification

`smoke_test_v9_gemma_1b.slurm` (job `133098`, N=20): **COMPLETED**, exit
`0:0`, 43:53 elapsed. Confirmed: no unexpected errors (only the expected
degenerate-calibration logs at tiny N), oracle field absent from the
thresholds JSON (removal verified in real output, not just code), leak-free
rewrite produces populated `final_answer` text, checkpoint counts correct
(80 calib + 80 test = 20 items × 8 steps).

## Suite launch status

**Status: SUBMITTED** — 2026-08-22, via `bash slurm/submit_v9_night.sh`.
Job ID → model/arm mapping follows the script's exact submission order.

| Job ID | Model | Arm | Status |
|---|---|---|---|
| 133620 | Llama-1B | baseline | |
| 133621 | Llama-1B | conformal | |
| 133622 | Llama-3B | baseline | |
| 133623 | Llama-3B | conformal | |
| 133624 | Gemma-1B | baseline | |
| 133625 | Gemma-1B | conformal | |
| 133626 | Gemma-4B | baseline | |
| 133627 | Gemma-4B | conformal | |
| 133628 | Phi-1.5 | baseline | |
| 133629 | Phi-1.5 | conformal | |
| 133630 | Phi-2 | baseline | |
| 133631 | Phi-2 | conformal | |

Fill in Status as jobs complete (`sacct -j <id> --format=JobID,State,ExitCode,Elapsed`).

## Prediction on record (before results land)

Based on retroactive analysis of V6 data with the folding-only fix applied
(`analyze_folding_only_definition.py`, α=0.10, global thresholds):
- **Phi-1.5, Phi-2**: expected to calibrate successfully (low fold rate
  regardless of setup).
- **Llama-1B, Llama-3B, Gemma-4B**: expected to still fail — confirmed via
  `analyze_data_needed.py` that their true fold rate exceeds the target at
  *every* threshold, a genuine behavioral ceiling, not a data-quantity
  issue. Fewer calibration items this run (150 vs. 250) if anything makes
  this harder, not easier.
- **Gemma-1B**: genuine toss-up — its one V6 success was oracle-assisted,
  and oracle access no longer exists in v9.

Once real results land, check them against this prediction rather than
assuming — that's the actual point of writing it down first.

## Results locations

`results/v9_night_medquad/run_baseline_<model>.json`,
`run_conformal_<model>.json`, `thresholds_<model>.json`. Test-phase
checkpoint: `<out>.calib.partial.jsonl` / `.test.partial.jsonl`.

## Next steps once complete

1. Check every thresholds file: real `tau_global` or still `-1.0`? Compare
   against the prediction above.
2. If some models newly calibrate, verify with real content (sample
   records, rewrite_triggered distribution), not just the exit code.
3. Feeds directly into the "how many models are reliable enough to
   produce a threshold" discussion for the writeup — see
   `MENTOR_DISCUSSION_2026-08-15.md` item #1 and the follow-up conversation
   on framing calibration failure as a disclosed finding vs. a weakness.
