# SycEval Team Update — 2026-08-09

*Follow-up to `docs/TEAM_UPDATE_2026-08-05.md` (the fix-campaign summary).
This one covers: the real V6 suite completing, what we found when we
actually looked at the results, and a real bottleneck we tracked down to
its root cause. Full detail: `docs/EXPERIMENT_TRACKER.md` (live job
status) and the `analyze_*.py` / `diagnose_*.py` scripts in the repo root
(each one documents its own method in its docstring).*

---

## TL;DR

The real N=1000 suite (36 jobs, full 2×2 matrix: MedQuad + HealthSearchQA
× oracle-on + oracle-off) **completed cleanly — zero crashes, zero
unexpected errors, zero judge parse failures.** But "completed cleanly"
turned out to only be step one. Digging into the actual calibration
results surfaced a real labeling bug (now fixed and validated) and — more
importantly — a genuine, now well-evidenced finding: **Llama (1B/3B) and
Gemma-4B fold to pressure at a rate too high to ever be statistically
certified as "safe," no matter how much calibration data or how loose a
target you use.** That's not a bug. That's a result.

---

## What ran

36 SLURM jobs, all `COMPLETED`, exit `0:0`, no exceptions:
- Baseline + CCR (both oracle settings) × 6 model families × 2 datasets
- N=1000 each (250 calibration / 750 test), α=0.05, group thresholds on
- Verified with `verify_v6_suite.py` — a script that scans all 36 outputs/thresholds/logs at once instead of checking by hand

**First finding from that scan**: calibration only succeeded in 2 of the
24 CCR runs at the original α=0.05 target. Everyone else defaulted to
"reject everything, rewrite every draft" — logged loudly (the C2 fix from
last time working as designed), not hidden, but clearly not the selective
risk-control system the methodology describes.

---

## Chasing down why — the bottleneck investigation

We didn't stop at "calibration mostly fails," because that's not
actionable on its own. Here's the chain of diagnosis, each step backed by
a script that reuses the *real* calibration math/data (no reruns, no
guessing):

1. **`analyze_alpha_sensitivity.py`** — relaxing the target from 5% to
   20% only rescues a few more models; most still fail even at 20%.
2. **`analyze_score_separation.py`** — checked whether the risk scorer
   even distinguishes bad outcomes from good ones. Answer: weak signal
   (AUC 0.55-0.72 everywhere), and both good AND bad outcomes get rated
   as high risk (mean scores 0.7-0.93) — a "ceiling effect."
3. **`diagnose_risk_scorer.py`** — pulled the scorer's actual reasoning
   text (the real run only ever saved the parsed number). Found the
   scorer is reasoning *correctly* — it was flagging **refusal under
   pressure** ("I can't help with this request") as risky. That's
   reasonable on its face, but:
4. **The actual bug**: our labeling logic (`classify_sychophancy`)
   silently treats "refused to answer" the same as "gave a wrong answer"
   — never a deliberate choice, just how a 3-way judge label happened to
   collapse. And structurally, if a model started out *wrong*, it could
   never be marked "bad" no matter what it did next, including refusing.
5. **Team decision**: refusing under pressure isn't "folding" — it's not
   great, but it's clearly better than actually adopting the wrong claim.
   We tested broadening "bad" to include refusal (`analyze_capitulation_bad_definition.py`)
   — that made things *worse* (broadening the target makes it harder to
   certify, obviously in hindsight). The fix that actually worked:
   **`analyze_folding_only_definition.py`** — narrow "bad" to require an
   actual correct→incorrect transition, with refusal explicitly excluded.
   Validated as a strict improvement with no downside (calibration
   successes went from 2→4 at 5%, 10→12 at 15%).
6. **The decisive check — `analyze_data_needed.py`**: for every model
   still failing even under the corrected definition, is it a data
   shortage (more calibration items would eventually fix it) or a
   genuine behavioral ceiling (the true fold rate is too high at *every*
   threshold, so no amount of data ever helps)? Answer: **11 of 12
   remaining failures are a hard ceiling — Llama-1B/3B and Gemma-4B
   fold too often, too consistently, for any amount of calibration data
   to certify a safe threshold.** Exactly one case (Gemma-1B,
   HealthSearchQA, oracle-on) is a genuine, trivially-fixable data
   shortage (needs ~1.4x more calibration items).

---

## The headline finding, stated plainly

**Llama-1B, Llama-3B, and Gemma-4B fold to adversarial pressure at a rate
that cannot be conformally bounded — not at 5%, not at 15%, not even at a
very loose 25%, and not with more data.** This isn't a pipeline failure —
it's a real, now rigorously-verified result that sharpens the project's
core "Confidence Trap" narrative with actual evidence instead of the
pre-fix pipeline's murky numbers.

Meanwhile Gemma-1B, Phi-1.5, and Phi-2 largely do calibrate successfully
under the corrected definition — the risk-control mechanism works, just
not universally.

---

## Bottlenecks / open decisions for the team

1. **Adopt the corrected bad-definition in the actual pipeline code.**
   Right now this fix only exists as a post-hoc analysis script — it
   hasn't been merged into `metrics.py`/`classify_sychophancy` for future
   runs. Low-risk, validated, just needs doing.
2. **Settle on a reporting alpha.** Given diminishing returns past ~15%
   and the credibility cost of a looser target, I'd recommend 10-15% as
   the defensible number, with per-model success/failure reported
   honestly rather than one blanket figure.
3. **Group thresholds should probably be dropped from headline numbers.**
   Nothing in this investigation suggests they're salvageable at
   N=1000/24-groups — that's a sample-size ceiling (per C4's own
   docstring: ~70+ items/group needed, we have ~10), not something the
   bad-definition fix touches.
4. **Decide whether to re-run test-phase decisions under the corrected
   definition.** Everything above is calibration-side analysis on
   already-collected data. The actual test-phase rewrite decisions in the
   36 completed jobs were made using the old (mostly `-1.0`,
   always-rewrite) thresholds. Adopting the fix properly means refitting
   thresholds and re-running test-phase decisions — real cluster time,
   not just analysis. Not started yet; wanted a team read before
   committing to it.
5. **The other items from the 2026-08-05 update are all still open**:
   HealthSearchQA data provenance, the shared-judge-model confound,
   the distillation cluster's fate, qualitative case studies, dependency
   pinning, and the abstract rewrite.

---

Happy to walk through any of the diagnostic scripts or the reasoning
transcripts directly — everything above is reproducible from data already
sitting in `results/`, nothing here required guessing.
