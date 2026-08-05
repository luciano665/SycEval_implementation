# SycEval Team Update — 2026-08-05

*For full technical detail: `review_findings.md` (the original audit),
`RESEARCH_WORKFLOW.md` (protocol/architecture), `docs/EXPERIMENT_TRACKER.md`
(live job status). This doc is the summary for everyone else.*

---

## TL;DR

The CCR pipeline had a set of real bugs that would have quietly invalidated
our results (silent oracle leakage, calibration/test scoring mismatches,
non-deterministic scoring, an ignored CLI flag, corrupted claim text, and
more). All of them are fixed and verified against real models. The real
N=1000 suite — both datasets, both oracle settings — is now running on the
HPC (36 jobs). First results are already landing.

---

## What we found

A full code review of the CCR pipeline and the existing V6 results turned
up 21 code-level findings and 7 issues in `abstract.md`'s claims. The
short version of why this mattered: **every existing result number we had
came from a pipeline with an undisclosed test-time oracle and several
calibration/scoring bugs.** The headline "Confidence Trap" finding
(filtering helps small models, hurts large ones) might still be real, but
we couldn't trust the specific numbers until this was addressed.

Some of the more consequential bugs, for context:

- The claim scorer, risk scorer, and rewriter were **silently** being
  shown the correct answer at test time — with zero disclosure anywhere in
  the code, output, or metadata. Every historical number was an
  oracle-assisted upper bound, not a deployable-system measurement.
- `--use_group_thresholds` was accepted on the CLI, recorded in run
  metadata as if honored, and then **ignored** by the actual code. Every
  V6 run that claimed "group-thresholds methodology" never used it.
- Calibration scored the *purified* draft while test scored the *raw*
  draft — different inputs to the same statistical procedure, which
  invalidates the risk-control guarantee even when calibration "succeeded."
- A claim-parsing bug ate leading numbers off medical claims (e.g. "500 mg
  daily..." → "mg daily...") — in a medical-claims pipeline, that's
  corrupting the actual content being evaluated.
- The risk scorer secretly ran at a hardcoded temperature regardless of
  config, and no RNG seed was ever set — results weren't reproducible
  run-to-run even with identical settings.

## What we fixed

13 of the code findings (the ones that would corrupt or invalidate data)
plus 3 more analysis/reporting bugs we knocked out along the way — all
committed on `cleanup-unused-files`, covered by 65 passing tests. Full
before/after breakdown for each is in the conversation log / commit
history if anyone wants the detail; happy to walk through any of them.

**Deliberately not fixed yet** (documented, not blocking):
- The statistical procedure behind threshold selection has a
  multiple-comparisons issue — the nominal α=0.05 guarantee is weaker than
  stated. Needs proper stats treatment, not a quick patch.
- Judge, rebuttal-generator, and risk-scorer are all the same model
  (Qwen2.5-7B-Instruct) — a self-grading confound worth disclosing.
- A handful of smaller reproducibility items (see "things to look into"
  below).

## How we verified the fixes actually work

Not just unit tests — we smoke-tested the real pipeline against real
models on the HPC before committing any real cluster time, covering every
major axis at small N (20 items): all 6 model families, both datasets
(MedQuad + HealthSearchQA), group thresholds on and off, oracle-truth on
and off. Everything came back clean — no crashes, and every
expected-to-degenerate-at-small-N case (calibration failing at N=10-20)
was caught and logged loudly instead of silently producing garbage.
Along the way we also caught two pure infrastructure issues (a model-path
symlink problem, an unrealistic SLURM time budget) that weren't pipeline
bugs at all, just environment setup gaps.

## Current status: real suite is running

36 jobs launched on `gpu_7day`, covering the full 2×2 design — both
datasets (HealthSearchQA, MedQuad) × both oracle settings (oracle-assisted
upper bound, and the real deployable no-oracle variant) × all 6 model
families, N=1000 each (250 calibration / 750 test), α=0.05, group
thresholds on.

Live job-by-job status, all 36 job IDs, results file locations: see
`docs/EXPERIMENT_TRACKER.md` in the repo. First 3 baseline jobs already
completed cleanly (12-19 hrs each); conformal jobs are slower (est.
40-50 hrs) and still running.

**Expect full completion within a few days**, all comfortably inside the
7-day SLURM budget per job.

---

## Things the team could look into

None of these block the current run — they're either follow-up work once
real numbers land, or things that could use a second pair of hands now:

1. **Statistical review of the threshold-selection method** — if anyone
   has conformal prediction / multiple-testing-correction experience,
   the current candidate-scanning procedure (`conformal_thresholds.py`)
   picks the best of many candidates without correcting for having tried
   many, which weakens the nominal confidence guarantee. Worth a proper
   fix (fixed-sequence testing or Learn-then-Test) before final writeup,
   or at minimum a precise statement of what guarantee we can actually
   claim.

2. **HealthSearchQA ground-truth provenance** — `data/healthsearch_qa.jsonl`
   is used as the reference answer everywhere (judge, claim scorer,
   rewriter). Its provenance isn't documented in the repo — if these
   "reference" answers are themselves model-generated rather than the
   official Med-PaLM release's ground truth, that needs disclosure in the
   paper. Worth someone tracking down how this file was originally built.

3. **Disjoint-judge experiment** — right now the same model
   (Qwen2.5-7B-Instruct) writes the rebuttals, judges correctness, *and*
   scores risk. Worth scoping whether it's feasible to rerun a subset with
   a different judge model to see how much this matters.

4. **Distillation cluster's fate** — `distill_eval.py`,
   `statistical_significance.py`, and 3 old distillation result files are
   still in the repo but not part of the current experiment line, and
   `statistical_significance.py` has a known unfixed pairing bug. Someone
   should decide: revive it, fix it, or formally retire it.

5. **Qualitative case studies** — `find_case_example.py` had a bug where
   it was silently reading placeholder thresholds instead of the real
   calibrated ones; that's fixed now, so it should actually work correctly
   for mining good example transcripts for the paper once real results are
   in. Could be a good task for someone to pick up in parallel with the
   suite finishing.

6. **Reproducibility cleanup** — unpinned dependencies in
   `requirements.txt`, no `--seed` CLI flag on `run_eval.py` (currently
   hardwired), and some cosmetic-but-confusing leftover SLURM log banners
   from old runs. Low priority, but good hygiene before anyone else tries
   to reproduce this.

7. **Abstract rewrite** — once real numbers are in, `abstract.md` needs a
   pass against the original review's findings (overclaiming "consistently
   reduces," hiding the oracle assumption, an unverifiable "7.4%" figure).
   Someone familiar with the intended narrative should own this rather
   than have it fall out of the technical fix work.

Happy to go deeper on any of these, or pair with whoever picks one up.
