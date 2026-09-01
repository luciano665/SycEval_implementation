# conformal_v10 Exact-CRC Suite Tracker

*Branch: `LM_corrections`. Companion to `V9_EXPERIMENT_TRACKER.md` (the
original v9n suite this is compared against) and
`conformal_v2/conformal_thresholds.py`'s `exact_crc_feasible` /
`fit_global_threshold_exact_crc` / `certifiable_fraction` docstrings for
the statistical rationale. Motivated by review_findings.md C14: the
existing Wilson-based `fit_global_threshold` scans many candidate tau
values, but a Wilson confidence interval is only valid for a single
evaluated candidate — an unresolved multiple-comparisons gap. This suite
swaps in the exact finite-sample bound (Angelopoulos & Bates, "Conformal
Risk Control") via a new `--threshold_method exact_crc` flag, which stays
valid under that same scan.*

---

## What's different from v9n

Everything else identical (deployable-only, leak-free rewrite,
folding-only labels, global thresholds, alpha=0.10, N=300 MedQuad,
calib_frac=0.5) — only the threshold-fitting bound changes:

| | v9n | v10 (this suite) |
|---|---|---|
| `--threshold_method` | (flag didn't exist; Wilson was the only option) | `exact_crc` |
| judge/rebuttal models | Qwen2.5-7B-Instruct | unchanged |
| baseline results | `results/v9_night_medquad/run_baseline_<model>.json` | **reused, not rerun** (see below) |

**No new baseline jobs, by default.** `run_eval.py` (baseline) never calls
`fit_thresholds`/threshold fitting at all — only the conformal arm does.
Since nothing about baseline generation changed, `run_baseline_<model>.json`
from v9n (same seed, same judge, same 300-item set) is the correct,
valid, already-computed baseline to pair every v10 conformal result
against. This suite is 6 jobs, not 12.

**If the v9 baseline data is ever unavailable** (deleted, not synced,
whatever): `submit_v10_suite.sh` checks for all 6
`results/v9_night_medquad/run_baseline_<model>.json` files before
submitting anything and aborts with a clear message if any are missing —
it will not silently leave you with conformal-only results that have
nothing to pair against. `slurm/v10_<model>_baseline.slurm` (6 scripts,
one per model) exist as an explicit fallback in that case — same
model/judge/data as v9n's baseline, just writing into
`results/v10_exact_crc_medquad/` instead. Not run by default; only submit
them if the preflight check in `submit_v10_suite.sh` fails.

## Local validation before spending cluster time

Before this doc existed: `fit_global_threshold_exact_crc` and
`certifiable_fraction` were unit-tested (`tests/test_selective_crc.py`,
9 tests) and the full `fit_thresholds`/CLI wiring was integration-tested
(`tests/test_calibration_flags.py`, 4 new tests covering: default stays
Wilson, exact_crc rescues a real small-n case, unknown method raises,
JSON round-trip carries the new fields). Full suite: 81/81 passing
locally. `analyze_selective_crc.py` was also proven end-to-end against a
synthetic fixture built to the real v9 checkpoint schema (three cases:
rescued / still-fails / both-certify, all produced correct verdicts) —
see conversation log / commit `28891d7` on `LM_corrections` for detail.

None of that used real model calls or GPU time — this tracker is for the
first run that does.

## Pre-launch verification (DO THIS FIRST)

Run `smoke_test_v10_gemma_1b.slurm` (N=20) before submitting the full
suite. Confirm:
- Exit 0, no crashes from the new `--threshold_method` argparse choice.
- `results/smoke_test_v10/smoke_thresholds_gemma_1b.json` contains
  `"threshold_method": "exact_crc"`, and `xi_certified`/`alpha_min` are
  real numbers, not null (confirms the diagnostic is actually being
  computed and saved, not silently skipped).
- `calibration_failed` at N=20 may still be `true` either way — expected
  at this tiny N (same caveat as the v9n smoke test), not a signal.

Smoke test job ID / status: **NOT YET RUN** — fill in once submitted.

## Suite launch status

**NOT YET SUBMITTED.** Run `slurm/submit_v10_suite.sh` after the smoke
test passes. Fill in job IDs below once submitted:

| Job ID | Model | Arm |
|---|---|---|
| — | Llama-1B | conformal (exact_crc) |
| — | Llama-3B | conformal (exact_crc) |
| — | Gemma-1B | conformal (exact_crc) |
| — | Gemma-4B | conformal (exact_crc) |
| — | Phi-1.5 | conformal (exact_crc) |
| — | Phi-2 | conformal (exact_crc) |

Check status: `sacct -j <ids> --format=JobID,JobName,State,ExitCode,Elapsed`

## Results locations

`results/v10_exact_crc_medquad/run_conformal_<model>.json`,
`thresholds_<model>.json`. Test-phase checkpoint:
`<out>.calib.partial.jsonl` / `.test.partial.jsonl`. Baseline: reuse
`results/v9_night_medquad/run_baseline_<model>.json` (see above).

## Prediction on record (before results land)

Based on `analyze_selective_crc.py`'s design and the reasoning in the
`LM_corrections` conversation log:

- **Phi-1.5, Phi-2**: expected unchanged (already certify under Wilson
  in v9n at tau_global=1.0 — a non-borderline success; exact_crc should
  not flip an already-easy case). Included as a control.
- **Llama-1B, Llama-3B, Gemma-4B**: expected to still fail. Their
  documented fold rate (~40%+) is far enough above alpha=0.10 that no
  valid statistical bound — Wilson or exact — can certify them; this
  isn't a bound-precision question, it's a true-rate ceiling
  (`analyze_data_needed.py`'s original finding). `alpha_min` should land
  near their true fold rate under either bound.
- **Gemma-1B**: genuine open question, not predictable in advance — it's
  the one model whose certification status has flipped across suite
  versions (v6 oracle-on succeeded, v9n deployable-only failed). If v9n's
  failure was partly a small-n Wilson artifact rather than a true-rate
  ceiling, this is the most likely case to flip to certified under
  exact_crc with the SAME calibration data, no more items needed.

Once real results land, check them against this prediction rather than
assuming — that's the actual point of writing it down first.

## Next steps once complete

1. Check every thresholds file: does `tau_global` change vs. the
   matching v9n file? Does `calibration_failed` flip for any model?
2. Compare `alpha_min` per model against `analyze_data_needed.py`'s
   original ceiling-vs-data-shortage verdicts for the corresponding
   models (different suite versions, so not a direct replication, but a
   sanity check: models flagged as hard ceilings there should show
   `alpha_min` well above 0.10 here too).
3. If Gemma-1B (or any other model) flips to certified, verify with real
   content (sample records, `rewrite_triggered` distribution) before
   reporting it as a genuine finding — same discipline as every prior
   suite in this project.
4. Feeds the "is calibration failure a real ceiling or a bound artifact"
   question directly — see the exchange on `LM_corrections` where this
   was scoped out.
