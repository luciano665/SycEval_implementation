# V6 Real-Suite Experiment Tracker

*Branch: `cleanup-unused-files`. Update this file as jobs land — it's meant
to be a living log, not a one-time snapshot. See `RESEARCH_WORKFLOW.md` for
the full pipeline/architecture writeup this tracker assumes you've read.*

---

## 1. The 2x2 design

Every model family runs against a baseline (no intervention) and CCR
(conformal claim refinement), on two datasets, with CCR run under both
oracle framings — whether the claim scorer / risk scorer / rewriter get to
see the reference answer (`--oracle_truth`, upper bound) or not
(`--no_oracle_truth`, the deployable variant).

| | HealthSearchQA | MedQuad |
|---|---|---|
| **Baseline** | `submit_v6_suite.sh` (part 1) | `submit_v6_suite_medquad.sh` (part 1) |
| **CCR, oracle-on** | `submit_v6_suite.sh` (part 2) | `submit_v6_suite_medquad.sh` (part 2) |
| **CCR, oracle-off** | `submit_v6_suite_no_oracle.sh` | `submit_v6_suite_medquad_no_oracle.sh` |

6 model families each: Llama-3.2 1B/3B, Gemma-3 1B/4B, Phi-1.5, Phi-2.
Judge + rebuttal model: Qwen2.5-7B-Instruct (all arms). N=1000 per run
(250 calibration / 750 test), alpha=0.05, group thresholds enabled.

**36 jobs total** across all 4 suites (12 + 6 + 12 + 6).

---

## 2. Pre-launch verification (smoke tests) — all passed

Before committing real cluster time, every major code path was exercised
against real models at small N (20 items) on the `cleanup-unused-files`
branch. All came back clean (no crashes; degenerate-calibration warnings
fired exactly where expected given the tiny sample):

| Job | What it validated | Result |
|---|---|---|
| 123393 | Gemma-1B, medquad, oracle-on, baseline+conformal | COMPLETED |
| 123566-123570 | Llama-1B/3B, Gemma-4B, Phi-1.5/2, medquad, oracle-on | COMPLETED, all 5 |
| 123747 | Gemma-1B, **healthsearch**, oracle-on, `--use_group_thresholds` | COMPLETED, tau_by_group populated correctly |
| 124210 | Gemma-1B, medquad, **--no_oracle_truth** | COMPLETED, oracle_truth:false confirmed in thresholds file |

Coverage: all 6 families, both domains, group thresholds, both oracle
settings — every dimension touched at least once.

---

## 3. Suite launch status

**Fill in job IDs and dates as you submit each suite.** Job ID → model
mapping follows script submission order (baselines then conformal, in
family order: llama_1b, llama_3b, gemma_1b, gemma_4b, phi_1.5, phi_2).

### HealthSearchQA — baseline + oracle-on (`submit_v6_suite.sh`)
**Status: SUBMITTED** — 2026-08-04

| Job ID | Model | Arm |
|---|---|---|
| 124249 | Llama-1B | baseline |
| 124250 | Llama-3B | baseline |
| 124251 | Gemma-1B | baseline |
| 124252 | Gemma-4B | baseline |
| 124253 | Phi-1.5 | baseline |
| 124254 | Phi-2 | baseline |
| 124255 | Llama-1B | conformal, oracle-on |
| 124256 | Llama-3B | conformal, oracle-on |
| 124257 | Gemma-1B | conformal, oracle-on |
| 124258 | Gemma-4B | conformal, oracle-on |
| 124259 | Phi-1.5 | conformal, oracle-on |
| 124260 | Phi-2 | conformal, oracle-on |

### HealthSearchQA — oracle-off (`submit_v6_suite_no_oracle.sh`)
**Status: NOT YET LAUNCHED**

| Job ID | Model | Arm |
|---|---|---|
| | Llama-1B | conformal, oracle-off |
| | Llama-3B | conformal, oracle-off |
| | Gemma-1B | conformal, oracle-off |
| | Gemma-4B | conformal, oracle-off |
| | Phi-1.5 | conformal, oracle-off |
| | Phi-2 | conformal, oracle-off |

### MedQuad — baseline + oracle-on (`submit_v6_suite_medquad.sh`)
**Status: NOT YET LAUNCHED**

| Job ID | Model | Arm |
|---|---|---|
| | Llama-1B | baseline |
| | Llama-3B | baseline |
| | Gemma-1B | baseline |
| | Gemma-4B | baseline |
| | Phi-1.5 | baseline |
| | Phi-2 | baseline |
| | Llama-1B | conformal, oracle-on |
| | Llama-3B | conformal, oracle-on |
| | Gemma-1B | conformal, oracle-on |
| | Gemma-4B | conformal, oracle-on |
| | Phi-1.5 | conformal, oracle-on |
| | Phi-2 | conformal, oracle-on |

### MedQuad — oracle-off (`submit_v6_suite_medquad_no_oracle.sh`)
**Status: NOT YET LAUNCHED**

| Job ID | Model | Arm |
|---|---|---|
| | Llama-1B | conformal, oracle-off |
| | Llama-3B | conformal, oracle-off |
| | Gemma-1B | conformal, oracle-off |
| | Gemma-4B | conformal, oracle-off |
| | Phi-1.5 | conformal, oracle-off |
| | Phi-2 | conformal, oracle-off |

---

## 4. How to check progress

```bash
# quick status of everything you've submitted
squeue -u $USER

# once a job finishes, check how it ended
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed

# check for genuine errors vs. expected degenerate-calibration ERROR logs
grep -inE "error|traceback|exited with" logs/error_<jobname>_<jobid>.txt \
  | grep -v "generation flags are not valid"
```

At N=1000 with 250 calibration items split across up to 24 groups (with
`--use_group_thresholds`), expect noticeably *less* degenerate calibration
than our N=20 smoke tests, but don't be surprised if some groups still
don't reach a valid tau — C4's own docstring estimates ~70+ clean items per
group are needed, i.e. several thousand calibration items for all 24 groups
to be well-estimated.

Expected runtime per job, extrapolated from smoke-test throughput: baseline
~17-30 hrs, conformal (calibration + test) ~40-50 hrs — comfortably inside
the 7-day SLURM budget on every script.

---

## 5. Results locations

| Dataset | Directory |
|---|---|
| HealthSearchQA | `results/healthsearch_v6_1000/` |
| MedQuad | `results/medDataset_v6_1000/` |

File naming within each: `run_baseline_<model>_v5.json`,
`run_conformal_<model>_v6.json` (oracle-on) or
`run_conformal_<model>_no_oracle_v6.json` (oracle-off), with matching
`thresholds_<model>*.json` files.

---

## 6. Once all 4 suites complete — analysis checklist

1. Re-run `fix_table3.py` per dataset (it already pairs baseline aggregation
   to the conformal test-split via `calibration_items` in metadata — C13).
2. Compare oracle-on vs oracle-off per dataset: this is the "oracle gap" —
   how much does removing ground-truth access from the CCR components cost.
3. Check every thresholds file's `calibration_failed` / `tau_claim_fallback`
   flags before trusting any run's numbers — don't assume N=1000 guarantees
   clean calibration everywhere.
4. Watch for `Judge parse failure` warning spikes in any job's error log —
   each one silently becomes `"erroneous"` (C10); a spike suggests a
   judge-model problem worth investigating before trusting that run's rates.
5. Re-run the paired-baseline comparison logic per oracle setting — don't
   compare oracle-off CCR numbers against the same baseline used for
   oracle-on (baseline itself doesn't change, but keep the comparison
   explicit in any table/plot).

---

## 7. Known, deliberately parked limitations

Not fixed, and not blocking this launch — documented here so they don't get
forgotten before writeup:

- **C14**: threshold-selection procedure has a multiple-comparisons /
  nested-data-reuse issue; the nominal alpha=0.05 guarantee is weaker than
  stated. Disclose in methods, don't silently claim the strict guarantee.
- **Shared judge/rebuttal/risk-scorer model**: all three roles use
  Qwen2.5-7B-Instruct. Self-grading confound; worth noting as a limitation.
- **C21 grab-bag**: no `--seed` CLI flag on `run_eval.py` (hardwired to 7),
  unpinned dependencies, misleading historical SLURM job echo strings
  (fixed going forward via this suite's scripts, but old result files may
  still have stale banners).
