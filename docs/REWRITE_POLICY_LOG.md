# Rewrite Policy Investigation Log

*Living doc — append as this investigation progresses. Companion to
`docs/RISK_SCORER_PROMPT_LOG.md` (a related but distinct thread: that one
is about the risk *scorer* prompt, which decides *whether* to trigger a
rewrite; this one is about the rewrite prompt/policy itself, which
decides *what happens* once triggered).*

---

## The finding

For models whose calibration fails (`tau_global = -1.0`), every response
still gets rewritten unconditionally — calibration failing means "no
certified guarantee," not "no intervention." `analyze_rewrite_effect.py`
checked, using only already-completed test-phase data (no new compute),
whether that unconditional rewriting actually helps or hurts, by
comparing `draft_sycophancy` (before rewrite) to `sycophancy` (after
rewrite) for every row where `rewrite_triggered == True`.

**Result — wildly different per model, not a uniform effect:**

| Model | Effect of rewriting | Consistency |
|---|---|---|
| Gemma-4B | Helps a lot (regressive rate drops 16-44 points) | All 4 configs (both datasets x both oracle settings) |
| Llama-3B | Hurts a lot (regressive rate rises 20-30 points) | All 4 configs |
| Llama-1B | Helps with oracle access, hurts without | Oracle-dependent |
| Gemma-1B, Phi-1.5, Phi-2 | Roughly neutral | Small effects either direction |

The naive aggregate ("rewriting helped on average," 0.666 -> 0.647 for
the three high-fold-rate models) is misleading — it's an average of two
large opposite effects (Gemma-4B strongly positive, Llama-3B strongly
negative) canceling out, not a real consistent pattern.

**Llama-3B's result independently replicates a pre-fix finding**: the
original code review (`review_findings.md`, finding A1) flagged Llama-3B
getting worse under the old, buggy pipeline. This confirms it's a real
model behavior, not a pipeline artifact.

## Why this happens — the mechanism

Traced `conformal_v2/safe_rewrite.py`'s `anti_sycophancy_rewrite()`
prompt directly. It takes the already-purified (safe-claims-only) draft
as input, but **re-shows the model the original rebuttal text** and asks
it to self-grade (`KEEP_DRAFT` / `REVERT_INITIAL` / `REVISE`). The prompt
is heavily biased toward resistance (default to REVERT_INITIAL, explicit
list of authority cues that don't justify a change) — but it's the same
`tested_model` that already caved once, not an independent reviewer.
Plausible explanation for the divergence: some models can use this
second, explicitly-framed pass to genuinely self-correct (Gemma-4B);
others cave to the same pressure again despite the framing (Llama-3B).

## Proposed fix directions (not yet acted on)

1. **Selective rewrite policy** — stop treating "rewrite everything that
   triggers" as one-size-fits-all; enable/disable per model based on
   measured effect. Cheapest option, no new prompt engineering.
2. **Reduce re-exposure to the rebuttal during rewrite** — restructure
   the prompt so it doesn't hand back the original persuasive framing.
   Directly targets the diagnosed mechanism. Cheap to test the same way
   the risk-scorer prompt was tested (small-N diagnostic first).
3. **Use a different (non-compromised) model for the rewrite step** —
   more expensive, and changes the framing from "self-correction" to
   "external correction," which is a methodology question worth
   discussing, not just an engineering one.

## Open methodological issue: how do we validate a selective policy without circularity?

Raised directly by the user, and it's a real, important catch: if we
decide "disable rewrite for Llama-3B" using the test-set finding above,
and then report "improved" numbers by applying that decision back onto
the *same* test set, that's circular — evaluating a policy on the exact
data that generated it doesn't prove anything.

**Why calibration data can't resolve this for the 36 already-completed
jobs**: `calibration_collect()` never calls the rewrite function at all
(its own docstring: "Calibration phase (NO rewriting)") — it only fits
tau from risk scores. So there is no calibration-phase signal about
whether rewriting helps; that question was only ever answered on test
data, because that's the only place rewriting happened.

**Two valid resolutions:**

- **Option A (proper fix, future runs only)**: extend
  `calibration_collect` to also test the rewrite's effect on a slice of
  calibration data — not to fit tau, but specifically to decide the
  rewrite-enable policy per model. Clean separation: calibration decides
  the policy, test data validates it honestly. Requires new compute on a
  future run.
- **Option B (usable now, zero new compute)**: split the *existing* test
  set itself — plenty of rows per model (6000 for Llama-3B/Gemma-4B).
  Randomly split into a "decision half" (used only to decide the policy)
  and a "held-out validation half" (used only to report the policy's
  actual effect, never touched during the decision). Legitimate
  train/validation split via resampling already-collected data.

**Status**: neither has been built yet. Option B is the natural next
step given it requires no new compute — decided but not yet implemented.

---

## Current state / next steps

1. Not yet decided which fix direction (1/2/3 above) to pursue.
2. Not yet built: the Option B split-based validation script, which
   would need to precede any "selective policy improves things" claim.
3. Regardless of which fix (if any) gets adopted, the finding itself —
   rewriting has strongly divergent, model-dependent effects, with
   Llama-3B's harm independently replicating a pre-fix finding — is
   real and worth reporting on its own merits.
