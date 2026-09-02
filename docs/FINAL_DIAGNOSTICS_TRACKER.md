# Final Diagnostics Tracker (pre-writing)

*Branch: `LM_corrections`. These are the two experiments requested in
mentor feedback (quoted below in full), explicitly framed as the last
ones before shifting into writing — not a new open-ended investigation.*

> Yeah, I think that framing makes sense. I'd probably avoid spending too
> much more time trying different thresholds or scoring models and
> instead characterize where the calibration is breaking. One useful
> experiment could be to test an oracle or ground truth risk scorer. If
> calibration works there, then the bottleneck is likely the risk scorer;
> if it still fails, that points to a more fundamental limitation. It may
> also be worth comparing directly against a rewrite only baseline to see
> how much of the sycophancy reduction is really coming from rewriting.
> After that, I'd lean toward shifting fully into writing.
> — Dr. Gyawali

---

## Diagnostic 1: oracle-assisted risk scorer (calibration-only)

**Question**: is calibration failure a risk-scorer limitation, or a
genuine behavioral ceiling in the tested model?

**Method**: `--oracle_risk_scorer_calib` (new flag, `run_conformal_v2.py`)
lets the risk scorer see the reference answer during CALIBRATION ONLY —
test time is untouched, stays leak-free (see `calibration_collect`'s
docstring). `mode=calibrate`, `--max_items 150` (matches v9n/v10's
calibration slice size exactly, so this isn't confounded by a larger
calibration set), `--threshold_method exact_crc`, `alpha=0.10`. Run for
the 4 models whose calibration currently fails: Llama-1B, Llama-3B,
Gemma-1B, Gemma-4B. Phi-1.5/2 already certify — nothing to diagnose.

**Prior evidence already on record, before running this**: the v3
risk-scorer prompt (`docs/RISK_SCORER_PROMPT_LOG.md`) raised Gemma-4B's
AUC from 0.60 to 0.849 — a large, real improvement in the scorer's
ability to rank bad vs. good outcomes — and recalibrating with that
better scorer (`calibrate_with_v3_scorer.py`) still failed identically
(bad rate 40.6%, fails at every alpha 0.05–0.25). That's strong prior
evidence against "scorer quality is the bottleneck." This diagnostic is
the cleanest, most decisive version of the same check: handing the
scorer the actual correct answer removes scorer-quality as a variable
entirely. If calibration still fails here, there is nothing further to
attribute to the risk scorer.

**Status**: NOT YET RUN. `slurm/submit_oracle_diagnostic.sh` (4 jobs).
Fill in job IDs once submitted:

| Job ID | Model | Result (tau_global) |
|---|---|---|
| — | Llama-1B | — |
| — | Llama-3B | — |
| — | Gemma-1B | — |
| — | Gemma-4B | — |

**Reading the result**: `results/oracle_diagnostic_medquad/thresholds_<model>.json`.
`tau_global` still `-1.0` (`calibration_failed: true`) even with oracle
access → confirms genuine behavioral ceiling, closes the "is it the
scorer" question for good. A real `tau_global` → the scorer WAS a real
bottleneck for that model, worth a follow-up (though this would be a
surprising reversal of the v3-scorer evidence above, and should be
checked against real content before reporting, not just the number).

**Do not use these thresholds operationally.** `oracle_risk_scorer_calib:
true` is recorded in the output for exactly this reason — there is no
matching oracle-assisted test-time path (deliberately; building one would
reintroduce the C1 leak the v9 deployable-only refactor removed).

---

## Diagnostic 2: rewrite-only comparison

**Question**: how much of any sycophancy reduction comes from the
rewrite step itself, versus the selective calibration/filtering on top
of it?

**Method, part A (zero new compute, done already)**: `analyze_rewrite_effect_v10.py`
reads `run_conformal_<model>.json`'s `individual_records` (already on
disk from v9n/v10) and compares `draft_sycophancy` (before rewrite) to
`sycophancy` (after) for every row where `rewrite_triggered == True`.
Repeats the method `docs/REWRITE_POLICY_LOG.md` used on the old,
pre-fix v6 data, on the current corrected pipeline instead.

**Key asymmetry, read this before interpreting output**: for the 4
models whose calibration fails, `tau_global = -1.0` means every test
draft gets rewritten — so for those models, **the existing conformal run
already IS the rewrite-everything policy**. The script's output for
Llama-1B/3B and Gemma-1B/4B directly answers the mentor's question with
no new experiment required. For Phi-1.5/2 (calibration succeeds),
rewrite is rare/selective — informative on its own, but doesn't show
what rewriting *everything* would do for those two models specifically.

**Method, part B (only needed for Phi-1.5/2, not yet built)**: if the
"what would rewrite-everything do to Phi models" question turns out to
matter for the writeup, it needs an actual new arm — a `--force_rewrite_all`
flag bypassing the threshold check in `test_apply`, then a `mode=test`
rerun (cheap: reuses the existing thresholds file, no recalibration)
for just those 2 models. Not built yet — only worth it if Diagnostic 2's
part A output makes it look decision-relevant.

**Status: RAN, real results, job 140665 (2026-09-02), against
`results/v9_night_medquad`:**

| model | rewrite_rate | regr_before | regr_after | delta | verdict |
|---|---|---|---|---|---|
| llama_1b | 1.000 | 0.133 | 0.121 | -0.012 | roughly neutral |
| llama_3b | 1.000 | 0.322 | 0.388 | +0.066 | HURTS |
| gemma_1b | 1.000 | 0.134 | 0.212 | +0.078 | HURTS |
| gemma_4b | 1.000 | 0.302 | 0.321 | +0.019 | roughly neutral |
| phi_1.5 | 0.000 | n/a | n/a | n/a | no rewrites triggered |
| phi_2 | 0.000 | n/a | n/a | n/a | no rewrites triggered |

**This overturns a previously-reported finding, worth flagging
explicitly.** `docs/REWRITE_POLICY_LOG.md` (computed on the old,
pre-fix v6 data, which had oracle-truth leaking into the rewrite prompt
— see `review_findings.md` C1) reported "Gemma-4B: Helps a lot
(regressive rate drops 16-44 points), all 4 configs." Under the
corrected, leak-free v9 pipeline, Gemma-4B's rewrite effect is
**roughly neutral, not a large help** (+0.019, near-zero, technically a
tiny regression). The most likely explanation: the old "helps a lot"
result was inflated by the rewrite prompt having test-time access to the
reference answer, which the v9 fix removed. Llama-3B's "hurts" finding
does replicate (+0.066 here vs. the earlier documented harm), so that
part was real, not an oracle artifact. Gemma-1B, previously characterized
as "roughly neutral" on old data, now shows a real, clear harm (+0.078).

**Bottom line for the writeup**: across the 4 uncertifiable models,
rewrite provides no clear benefit anywhere, and actively hurts in 2 of 4
(Llama-3B, Gemma-1B) under the corrected, leak-free pipeline. Combined
with calibration's -1.0 sentinel meaning "rewrite everything" is already
the deployed policy for these models, this is a second independent
negative result: not only can these models not be certified safe, the
mitigation step you'd fall back to doesn't reliably help either. That's
a stronger, cleaner version of the "Confidence Trap" narrative than what
was in the pre-fix data.

---

## After both land

Per the feedback above: write these two results up, then move into the
writing phase. Not a cue to chase further threshold/scorer/data variants
— that door is closed by the evidence already in `docs/RISK_SCORER_PROMPT_LOG.md`,
`analyze_data_needed.py`, and the exact-CRC robustness check
(`docs/V10_EXPERIMENT_TRACKER.md`).
