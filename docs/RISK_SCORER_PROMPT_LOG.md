# Risk Scorer Prompt Iteration Log

*Living doc — append a new section each time a prompt version is tested.
Purpose: keep the actual evidence in one place instead of scattered across
conversations. Companion to `docs/EXPERIMENT_TRACKER.md` (job tracking) and
the `diagnose_risk_scorer*.py` scripts (the actual test harnesses — read
their docstrings for full methodology).*

---

## Why this thread started

`analyze_score_separation.py` (full 24-run analysis) found the risk
scorer has a ceiling effect: it rates almost everything as high-risk,
good and bad outcomes alike (mean scores 0.7-0.93), giving weak AUC
(0.55-0.72) everywhere. `diagnose_risk_scorer.py` pulled the scorer's
actual reasoning text and found it was reasoning coherently — it was
correctly flagging refusal-under-pressure as risky, which is defensible,
but our own "bad" label doesn't consider refusal bad (see
`analyze_folding_only_definition.py` / the `bad_v4` decision). That
suggested a prompt-criteria problem, not a scorer-capability problem —
worth testing before considering a stronger/different model.

---

## OLD (production prompt, unchanged baseline)

Location: `conformal_v2/syco_risk.py`, `sycophancy_risk_score()`.
No calibration anchors, binary "changed under pressure = high risk"
criterion, no distinction between refusing and actually agreeing with the
rebuttal's claim.

Baseline numbers (N=120 fresh items, Llama-3.2-1B, healthsearch,
oracle-on): AUC 0.490 (no better than chance in this sample), refusals
scored >=0.7: 49/55 (89%).

---

## v2 — calibration anchors + maintained/declined/adopted bands

**Change**: added explicit instruction to use the full 0-1 range, and a
3-way classification (a=maintained, b=declined, c=adopted) with score
bands per category, so refusal (b) is no longer automatically lumped
with genuine wrong answers (c). Script: `diagnose_risk_scorer_v2.py`.

**Result (Llama-3.2-1B, N=120, same items as OLD baseline)**:

| metric | OLD | v2 |
|---|---|---|
| mean score, bad | 0.864 | 0.705 |
| mean score, good | 0.848 | 0.611 |
| gap (bad-good) | 0.016 | 0.094 |
| AUC | 0.490 | 0.599 |
| refusals scored >=0.7 | 49/55 | 5/55 |

**Verdict**: real improvement on the targeted symptom (refusal ceiling
effect, 89% -> 9%), and AUC/gap both improved. **New problem found**:
per-instance inspection showed some genuine folds (draft_label ==
"incorrect", a real wrong answer, not a refusal) got miscategorized into
the mid-range "declined" band instead of high-risk "adopted" — band (b)
was catching "doesn't sound like explicit agreement" when it should only
catch true non-answers.

---

## v3 — sharpened (b)/(c): true non-answer vs. any substantive wrong claim

**Change**: (b) narrowed to require a TRUE non-answer (no substantive
claim at all); (c) broadened to explicitly cover any substantive wrong
claim "regardless of whether it echoes the rebuttal's wording." Script:
`diagnose_risk_scorer_v3.py`, tested on two models to check both "did the
fix work" and "does it generalize."

**Result — Llama-3.2-1B (N=120, same items, repeat of the v2 test)**:

| metric | OLD | v2 | v3 |
|---|---|---|---|
| mean score, bad | 0.864 | 0.705 | 0.686 |
| mean score, good | 0.848 | 0.611 | 0.552 |
| gap (bad-good) | 0.016 | 0.094 | 0.134 |
| AUC | 0.490 | 0.599 | 0.638 |
| refusals scored >=0.7 | 49/55 | 5/55 | **0/55** |
| genuine folds scored <0.6 (miscategorized) | 3/28 | 14/28 | 15/28 |

**Result — Gemma-3-4B (N=120, new model, generalization check)**:

| metric | OLD | v2 | v3 |
|---|---|---|---|
| mean score, bad | 0.938 | 0.961 | 0.960 |
| mean score, good | 0.875 | 0.921 | 0.784 |
| gap (bad-good) | 0.062 | 0.039 | 0.176 |
| AUC | 0.604 | 0.654 | **0.849** |
| refusals scored >=0.7 | 7/7 | 4/7 | 3/7 (n=7, small sample) |
| genuine folds scored <0.6 (miscategorized) | 0/61 | 0/61 | 0/61 |

**Verdict**:
- Llama-1B: refusal-ceiling problem now fully resolved (0/55). The
  targeted fix (sharpening b/c wording) did **not** fix the
  fold-miscategorization issue — stayed ~50% (14/28 -> 15/28), essentially
  unchanged. The wording change didn't shift the model's behavior on
  those specific ambiguous cases; hypothesis about the exact cause was
  incomplete. Not yet understood why these particular cases resist
  reclassification.
- Gemma-4B: never had the fold-miscategorization problem. AUC jumped to
  0.849 — stronger separation than any of the 24 models in the original
  full-scale analysis (which topped out ~0.72). This is the most
  promising result so far: Gemma-4B was one of the models flagged as
  "no amount of data can fix this" (`analyze_data_needed.py`), and this
  level of separation improvement could plausibly change that verdict.

---

## Current status / open threads

1. **RESOLVED — Gemma-4B calibration test with v3 scoring: still fails.**
   Ran `calibrate_with_v3_scorer.py` (real `calibration_collect()`, v3
   scorer swapped in via verified monkeypatch, N=60 items / 480 records,
   healthsearch, oracle-on). Result: `bad_v4` rate = 40.6%, `fit_global_threshold`
   FAILS at every alpha tested (0.05 through 0.25). This matches the
   original full-scale finding for this exact config (41.9% in the real
   36-job suite), confirming the ~40% fold rate is real and consistent,
   not a small-sample fluke. **Conclusion: the AUC improvement from v3
   (0.60 -> 0.85 on the small N=120 sample) does not translate into
   calibration success — a better-ranking scorer cannot manufacture
   safety in a population where ~40% of items are genuinely bad.** This
   confirms `analyze_data_needed.py`'s original verdict: Gemma-4B's
   failure is a genuine behavioral ceiling, not fixable via scorer
   quality, more data, or a looser alpha. **This closes the
   prompt-engineering thread for Gemma-4B specifically** — no further
   scoring-side iteration is expected to help this model.
2. **Llama-1B's fold-miscategorization issue is still unexplained** —
   worth pulling the actual reasoning text for a few of the persistent
   cases (the ones stuck at ~0.50 across both v2 and v3) to understand
   why sharper wording didn't change the outcome, if it becomes a
   priority again. Lower priority now that Gemma-4B (the more promising
   lead) has a conclusive negative result — Llama-1B's own fold rate
   was already known to be too high to rescue regardless
   (`analyze_data_needed.py`).
3. Nothing here has been merged into `conformal_v2/syco_risk.py` yet —
   all three versions (OLD/v2/v3) currently coexist only as separate
   functions across `syco_risk.py`, `diagnose_risk_scorer_v2.py`, and
   `diagnose_risk_scorer_v3.py`. v2's refusal-mislabeling fix is still a
   real, validated improvement worth adopting on its own merits (cleaner
   scores, matches the `bad_v4` philosophy) even though it doesn't rescue
   Gemma-4B or Llama-1B's calibration outcomes.
