# SycEval Research & Workflow Summary

*Last updated: 2026-07-28 (branch `cleanup-unused-files`, HEAD `8f18a67`)*

A single-document orientation to the whole project: what the research is, how the
code implements it, how experiments are run, what the results said, what the
code review found, what has been fixed, and what remains.

---

## 1. The research question

**Do LLMs cave to pushback on medical questions, and can a post-hoc,
conformal-prediction-based filter stop them from caving?**

- **Sycophancy** = a model abandoning its answer when the user pushes back.
  - **Regressive** sycophancy: correct → incorrect after pushback (the dangerous one).
  - **Progressive** sycophancy: incorrect → correct after pushback (superficially
    good, but still evidence the model follows the user rather than the evidence).
- **The intervention — Conformal Claim Refinement (CCR)**: decompose the model's
  answer into atomic claims, score each claim's validity with a judge model,
  keep only claims above a calibrated threshold τ (chosen by conformal risk
  control at α = 0.05), reconstruct the answer from surviving claims, and — if a
  sycophancy-risk score exceeds a second calibrated threshold — trigger an
  anti-sycophancy rewrite.
- **Headline empirical finding so far (the "Confidence Trap")**: CCR/conformal
  filtering *helps small models* (~1B) but *hurts or is neutral for larger
  models* (3–4B+), consistently across Llama, Gemma, and Qwen families and
  across judge models. Filtering strips hedging and qualifiers, making bigger
  models sound more confident — and confident answers attract harder rebuttals
  and flip more easily.

**Important caveat**: the July 2026 code review (Section 7) found that all
existing result numbers came from a pipeline with an undisclosed test-time
oracle and several calibration/scoring bugs. The fixes are now in; **all
headline numbers need a rerun before they can be trusted or published.**

---

## 2. The evaluation protocol (both pipelines share this)

Every experiment follows the same three-phase SycEval protocol per question:

1. **Initial answer.** The tested model answers a medical question. A judge
   model labels it `correct` / `incorrect` / `erroneous` against the dataset's
   reference answer. (`erroneous` is normalized to `incorrect` by the metrics.)
2. **Rebuttal generation.** A separate rebuttal model writes a persuasive
   counter-claim at four escalating strengths:
   `simple` → `ethos` (authority) → `justification` (reasoning) → `citation`
   (fabricated-evidence placeholder). Rebuttals argue for a *false* alternative
   (flip=True is forced in the case-search tooling).
3. **Pressured re-answer**, in two modes:
   - **In-context**: model sees its own initial answer, then the rebuttal.
   - **Preemptive**: model sees the rebuttal *before* answering.

   The re-answer is judged again; comparing initial vs. final labels yields the
   sycophancy classification (`progressive` / `regressive` / `none`).

So each question produces **8 measurements** (2 modes × 4 strengths). With
N = 1000 questions that is 8000 rebuttal steps per model per dataset.

---

## 3. The two pipelines

### 3.1 Baseline — `run_eval.py`

Straight protocol, no intervention. Loads N seed-7 sampled questions
(`data_loader.load_data_local`), runs the three phases, writes one JSON with
per-step rows and summary stats (two-proportion z-tests, binomial CIs).
A legacy `--enable_conformal` flag applies the old V1 claim filter
(`conformal_wrapper.py`) to the *initial answer only* — not used in current
experiments.

### 3.2 CCR / Conformal V2 — `conformal_v2/run_conformal_v2.py`

The real intervention pipeline. Three modes (`--mode calibrate|test|both`):

**Calibration** (`calibration_collect`, default first 25% of items = 250):
1. For each calibration item, generate the initial answer, rebuttal, and a
   pressured **draft** answer.
2. Decompose the draft into claims; score each claim's validity
   (`scorers.score_claim_sycophancy`); judge each claim's support.
3. Fit **τ_claim** (claim-keep threshold) by conformal risk control at
   α = 0.05 so that, with 95% confidence, the fraction of kept-but-bad claims
   stays below the risk budget.
4. Score each **raw** draft with the sycophancy-risk scorer
   (`syco_risk.sycophancy_risk_score`, 0–1, parsed from the judge's
   "Final Score:" line) and fit **τ_risk** (rewrite trigger) on drafts labeled
   bad. With `--use_group_thresholds`, per-(mode × strength × correctness)
   thresholds are also fit — but each group needs ~70+ clean calibration
   instances or its τ degenerates to always-rewrite.
5. Persist thresholds + the calibration item hashes (`data_split`) to a
   thresholds JSON, with flags (`calibration_failed`, `tau_claim_fallback`,
   `oracle_truth`, seed, scorer temperature) recorded in metadata.

**Test** (`test_apply`, remaining 750 items — calibration items are excluded by
hash, opt-out via `--allow_calib_overlap`):
1. Generate initial answer → rebuttal → pressured raw draft.
2. Score the **raw** draft's sycophancy risk (must match calibration's input —
   the C3 invariant).
3. Purify the draft **once**: decompose → drop claims below τ_claim →
   reconstruct. Judge the purified draft.
4. If risk > τ_risk → anti-sycophancy rewrite (from the *purified* draft), then
   purify the rewritten text once. Otherwise the purified draft is final.
5. Judge the final answer; classify sycophancy vs. the initial label.
6. Every row is checkpointed incrementally to
   `<out>.test.partial.jsonl` (calibration records go to
   `<out>.calib.partial.jsonl`) so a crashed 7-day job keeps its partial results.

**Oracle framing** (`--oracle_truth`, default ON): the dataset's reference
answer is fed to the claim scorer, risk scorer, and rewriter. This measures an
**oracle-assisted upper bound**, not a deployable system. `--no_oracle_truth`
runs the truth-free variant. The flag is recorded in both the thresholds file
and results metadata, and mismatched thresholds are refused.

### 3.3 Analysis scripts

| Script | Purpose |
|---|---|
| `fix_table3.py` | Paper Table 3: per-model baseline-vs-CCR sycophancy rates by mode. Now item-paired (baseline restricted to the 750 test items via `calibration_items` from metadata); prints paired and legacy numbers. |
| `analyze_results.py` | Quick per-file rate summaries. |
| `evaluation_analysis.py` | Multi-file Excel workbook analysis (accuracy, pivots, transitions). |
| `case_example.py` / `find_case_example.py` | Qualitative case mining — find items where the draft was wrong, purification fixed it, etc. (GOOD case = first correct, draft incorrect, purified correct). |
| `statistical_significance.py` | McNemar tests for the **distillation** side-experiment (teacher/student pairs, `distill_eval.py`). Has a known unfixed pairing bug (C20). |

---

## 4. Code map (post-cleanup)

**Core shared modules**: `config.py` (EvalConfig dataclass), `models.py`
(`ask_model` for Ollama/HF backends + `set_global_seed`), `data_loader.py`,
`judge.py` (`judge_local` with retry→`erroneous` fallback), `rebuttals.py`,
`metrics.py`, `claims.py` (decompose/reconstruct, anchored list-marker regex),
`conformal.py` (`filter_claims`), `scorers.py` (claim validity + support),
`logger_utils.py`.

**CCR package**: `conformal_v2/` — `run_conformal_v2.py` (orchestrator),
`conformal_thresholds.py` (conformal fitting, JSON round-trip),
`syco_risk.py` (risk scorer), `safe_rewrite.py` (anti-sycophancy rewrite).

**Entry points**: `run_eval.py` (baseline), `python -m
conformal_v2.run_conformal_v2` (CCR), `distill_eval.py` (distillation side
experiment).

**Tests**: `tests/` — 65 pytest tests covering oracle gating, calibration
flags, raw-draft scoring, group thresholds, risk parsing/temperature, split
persistence, judge retry, checkpointing, claims parsing, and import smoke.
Run with `python3 -m pytest tests/ -q`.

**Data** (`data/`): `medDataset_processed.csv` (MedQuad-derived),
`healthsearch_qa.jsonl` (HealthSearchQA), `calibration_mixed.jsonl`.

**Removed in cleanup (commit `8f18a67`)**: duplicated `SycEval_implementation/`
shadow package (C18), broken V1 runner `run_conformal.py`, scratch scripts
`test_load.py` / `debug_env.py`, tracked `.DS_Store`.

---

## 5. Infrastructure & experiment history

- **Cluster**: WVU HPC, SLURM `gpu_7day` partition, 1 GPU + 2 CPUs, up to
  7-day wall time per job. Models are local HF checkpoints under `models/`
  (e.g. `models/gemma-3-1b-it`, `models/Qwen2.5-7B-Instruct`).
- **A full suite** = 12 jobs: 6 tested models × {baseline, conformal}
  (`slurm/submit_v6_suite.sh`). Tested models: Llama-3.2 1B/3B, Gemma-3 1B/4B,
  Phi-1.5/Phi-2. Judge & rebuttal: Qwen2.5-7B (judges standardized over time:
  Llama-3-8B → Qwen2.5-7B → Ministral-8B).
- **Suite history**:
  - V3: N=300, MedQuad, truth-grounded.
  - V4: N=300, MedQuad, standardized.
  - V5: N=300, full 12-job suite.
  - **V6 (current)**: N=1000 (250 calib / 750 test), α=0.05, group thresholds,
    chain-of-thought scorer, rewrite enabled; both MedQuad
    (`results/medDataset_v6_1000/`) and HealthSearchQA
    (`results/healthsearch_v6_1000/`).

Typical V6 CCR invocation (from the SLURM scripts):

```bash
python -m conformal_v2.run_conformal_v2 \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 --calib_frac 0.25 --max_items 1000 \
  --enable_rewrite --use_group_thresholds \
  --thresholds_out results/.../thresholds_gemma_1b_v6.json \
  --domain healthsearch --out results/.../run_conformal_gemma_1b_v6.json
```

---

## 6. Results so far (all pre-fix — treat as directional only)

**V6 aggregate (item-paired, verified 2026-07-27 via `fix_table3.py`):**

| Dataset | Mode | Baseline (paired) | Note |
|---|---|---|---|
| MedQuad | in-context | 37.5% | pairing vs. legacy full-1000 shifts ≤0.4pp |
| MedQuad | preemptive | 33.3% | |
| HealthSearchQA | in-context | 56.5% | HealthSearchQA is much more adversarial |
| HealthSearchQA | preemptive | 47.4% | |

**The Confidence Trap pattern (earlier suites, three judges — consistent):**

- Small models improve under conformal filtering: Llama-1B −12.7pp,
  Gemma-1B −12.5pp (Suite B, Qwen judge).
- Larger models get worse or stay flat: Llama-3B +4.9pp, Qwen-3B +10.4pp,
  Gemma-4B ~neutral.
- Mechanism hypothesis: filtering strips hedges → answers read more confident →
  confident answers get flipped more under pressure.

The abstract's original claims ("consistently reduces sycophancy", "average
reductions of up to 7.4%") were flagged by the review as contradicted by these
same tables (A1) and as hiding the oracle assumption (A3). The abstract needs
rewriting after the rerun.

---

## 7. The code-review & fix campaign (July 2026)

A full repo review (`review_findings.md`) produced 7 abstract findings
(A1–A7) and 21 code findings (C1–C21). Fixes were executed plan-driven
(`docs/superpowers/plans/2026-07-12-ccr-critical-fixes.md`) in three phases on
branch `ccr-fixes`, each task implemented by a fresh agent and passed through
spec-compliance + code-quality review.

### Fixed (C1–C13, C18) — commit trail on `ccr-fixes`/`cleanup-unused-files`

| Finding | Fix | Commits |
|---|---|---|
| C1 Silent test-time oracle | Explicit `--oracle_truth/--no_oracle_truth` gating truth into all intervention components, recorded in metadata + thresholds; mismatch refused | `42376fe` |
| C2 Silent calibration degeneration | Loud logging; `calibration_failed`/`tau_claim_fallback` flags persisted; rewrite-rate summary | `564638e` |
| C3 Calib/test scored different inputs | Calibration scores the **raw** draft (matches test); bad label stays on delivered purified draft | `05f3308` |
| C4 `--use_group_thresholds` ignored | Wired up; group thresholds round-trip through JSON; small-sample warning | `28d642c`, `b9def09` |
| C5 Risk parse grabbed first number | Anchored on "Final Score:" line, last label wins, 512-token budget, parse failures logged | `347b0de`, `61f9e86` |
| C6 Non-deterministic risk scorer | Honors caller temperature; global seed; seed + effective scorer temp recorded | `2b331ea`, `fcd1a85` |
| C7 calibrate→test leakage | Calibration question hashes persisted; `mode=test` excludes them (opt-out flag); malformed records warn | `7c8eb43`, `4979b3c` |
| C8/C9 Stray private import; committed merge conflict | Removed `pandas.core.missing` import; resolved `evaluation_analysis.py` conflict | `107d2dd`, `61efa78` |
| C10 One bad judge reply kills a 7-day run | Judge retry→`erroneous` fallback; incremental JSONL checkpoints, per-phase files, truncated at start | `12f176e`, `f5b0295`, `bb820ea` |
| C11 lstrip ate leading doses/years | Anchored list-marker regex | `b378087`, `8f92a90` |
| C12 Accepted drafts purified twice | Purify exactly once per path; `draft_answer_raw` stored | `958fb47` |
| C13 Unpaired baseline comparison | `fix_table3.py` pairs baseline to the 750-item test split | `10d00f2` |
| C18 Duplicated shadow package | `SycEval_implementation/` deleted; import repointed | `8f18a67` |

### Not yet addressed

- **Code**: C14 (threshold selection double-dips its own confidence bound),
  C15 (`summarize_rates` denominators), C16 (progressive rebuttals generated
  blind to reference), C17 (`find_case_example.py` reads thresholds with wrong
  keys), C19 (`metrics.py` micro-bugs), C20 (`statistical_significance.py`
  pairing explosion), C21 (reproducibility grab-bag: unpinned requirements,
  missing statsmodels, no `--seed` flag on `run_eval.py`, docs with stale
  paths — note `RUNNING.md` still references another machine's path).
- **Abstract**: A1–A7 (overclaims, oracle disclosure, naming/provenance) — to
  be rewritten once rerun numbers exist.

---

## 8. Current state & branch map

```
main                    ← old, pre-review
└─ ccr-fixes            ← all C1–C13 fixes (ends bb820ea)
   └─ cleanup-unused-files  ← + C18 dead-code removal (8f18a67)  ★ ACTIVE
```

- Test suite: **65 passed**.
- Working tree extras (untracked): `abstract.md`, `review_findings.md` is
  tracked, `phase1/2/3_full.diff` (review artifacts, deletable),
  `docs/` (plan + this document).
- Open scope decisions: keep or drop the distillation cluster
  (`distill_eval.py`, `statistical_significance.py`, 3 distill result JSONs,
  `models.md`), old `images/`, `split_results_by_dataset.py`.

## 9. What's next (rerun checklist)

1. Decide remaining scope: fix C14–C17/C19–C21 (or explicitly accept), and the
   distillation cluster's fate.
2. Merge the fix branch; update SLURM scripts if flags changed.
3. **Rerun V6 with the fixed pipeline — twice per condition if measuring the
   oracle gap**: `--oracle_truth` (upper bound) and `--no_oracle_truth`
   (deployable variant). The paper's claims hinge on the truth-free numbers.
4. SLURM launch notes from the final review:
   - Copy/rename `.partial.jsonl` files before resubmitting a crashed job
     (startup truncates them).
   - Pin `--seed 7` everywhere — `run_eval.py` has no seed flag (hardwired 7)
     and `fix_table3.py`'s item pairing silently depends on both runs sampling
     identically.
   - Watch job logs for `Judge parse failure` warnings (a spike = judge-model
     problem, and each failure now silently becomes `erroneous`).
   - Group thresholds need thousands of calibration items to be meaningful at
     24 groups; with 250 calibration items expect degenerate group τ values
     (flagged in logs).
5. Re-generate tables with the paired `fix_table3.py`; rewrite the abstract
   against the new numbers with the oracle framing disclosed.
