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

**Status: ALL 12 COMPLETED** — submitted 2026-08-22, all confirmed
`COMPLETED` exit `0:0` as of 2026-08-23. Job ID → model/arm mapping
follows the script's exact submission order.

| Job ID | Model | Arm | Elapsed |
|---|---|---|---|
| 133620 | Llama-1B | baseline | 03:31:07 |
| 133621 | Llama-1B | conformal | 07:25:38 |
| 133622 | Llama-3B | baseline | 05:41:49 |
| 133623 | Llama-3B | conformal | 12:14:46 |
| 133624 | Gemma-1B | baseline | 05:45:30 |
| 133625 | Gemma-1B | conformal | 12:13:52 |
| 133626 | Gemma-4B | baseline | 07:26:01 |
| 133627 | Gemma-4B | conformal | 13:19:08 |
| 133628 | Phi-1.5 | baseline | 06:16:51 |
| 133629 | Phi-1.5 | conformal | 10:39:47 |
| 133630 | Phi-2 | baseline | 07:25:04 |
| 133631 | Phi-2 | conformal | 11:40:03 |

**Confirms the run finished — not yet whether the content is trustworthy.**
See "Next steps" below before treating any thresholds/results as final.

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
