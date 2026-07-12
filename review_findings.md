# Repo Review Findings

Scope: `abstract.md` + the CCR pipeline (`conformal_v2/`, supporting root modules, SLURM suite, analysis scripts) and the v6 results in `results/medDataset_v6_1000/` and `results/healthsearch_v6_1000/`.
No code was changed; this file only documents findings. Ordered most-critical-first within each section.

Numbers below were recomputed from the `statistical_results` blocks of the 24 v6 run files (6 models × baseline/CCR × 2 datasets).

---

## Part 1 — Abstract issues (`abstract.md`)

### A1. CRITICAL — "consistently reduces" is contradicted by the repo's own results

> "we show that CCR consistently reduces sycophancy rates"

Recomputed from the v6 result files (overall sycophancy, baseline → CCR):

| Model | MedQuAD | HealthSearchQA |
|---|---|---|
| Llama-3.2-1B | 42.0% → 40.4% (−1.6) | 60.9% → 57.7% (−3.2) |
| **Llama-3.2-3B** | **43.6% → 49.9% (+6.3)** | **56.0% → 73.9% (+17.8)** |
| Gemma-3-1B | 36.3% → 33.6% (−2.7) | 58.1% → 51.2% (−6.8) |
| Gemma-3-4B | 49.0% → 42.9% (−6.1) | 72.2% → 31.6% (−40.6) |
| Phi-1.5 | 17.8% → 16.3% (−1.5) | 27.6% → 26.2% (−1.4) |
| Phi-2 | 25.1% → 20.9% (−4.1) | 36.2% → 26.3% (−9.9) |

- Llama-3.2-3B gets **worse on both datasets** — regressive sycophancy jumps 34.6%→45.6% (MedQuAD) and 49.1%→**71.7%** (HealthSearchQA). That is not a rounding-level exception; it's the largest single effect in the table after Gemma-4B's improvement.
- Phi-2's regressive rate on HealthSearchQA also ticks up (15.61%→15.68%).
- The MedQuAD **preemptive** macro-average worsens under CCR (33.7%→34.4%).
- This is the same "Confidence Trap" pattern the repo itself documents in `conformalResults.md` ("Large Models (3B+): CP often **worsens** sycophancy"). The abstract now claims the opposite without explaining what changed.
- The abstract also contradicts itself: "consistently reduces sycophancy rates" (sentence 5) vs. "lowers sycophancy in most settings" (sentence 6). Both can't be right; the data supports only the weaker one.

**Fix:** Replace "consistently reduces" with an accurately scoped claim, e.g. "reduces average sycophancy rates on both datasets, though per-model effects vary and Llama-3.2-3B worsens under CCR." Either present the Llama-3B regression as a finding (it connects to the Confidence Trap narrative) or explain why it is excluded.

### A2. CRITICAL — "conformal risk control" describes machinery that degenerated in every reported run

> "applies claim-level conformal risk control to filter high-risk statements before reconstructing the final response through a rewrite intervention"

What the saved thresholds files show for all 12 v6 runs:

- `tau_claim = 0.3` in **all 12** thresholds files. 0.3 is the hardcoded fallback constant in `conformal_v2/run_conformal_v2.py:378` used when claim-level calibration finds **no valid threshold** at `claim_alpha=0.05`. Twelve independent calibrations landing on exactly 0.30 is only plausible via the fallback path — i.e., the claim-level "conformal" threshold was never actually calibrated; runs used an arbitrary constant. (Verify against the run logs, which would contain the `"Claim-level threshold calibration found no valid tau"` warning.)
- `tau_global = -1.0` in **10 of 12** runs (all HealthSearchQA + Llama/Gemma MedQuAD). Since the rewrite rule is `s > tau`, tau = −1.0 means **every single test draft was rewritten**. In those runs there is no selective risk control at all — the system is "purify with a fixed threshold + always rewrite." Only the two MedQuAD Phi runs got a real threshold (0.9).
- The α = 0.05 target (regressive rate among accepted drafts ≤ 5%) was therefore never certified — calibration failed to find any threshold achieving it — and final regressive rates in the outputs range from 7% to 72%.

**Fix:** Either (a) fix the calibration so it succeeds and report coverage/violation statistics, or (b) describe the method honestly (e.g., "a claim-filtering + rewrite pipeline; conformal calibration was attempted but the risk target was unattainable at α=0.05, so the deployed policy rewrote all drafts in 10/12 configurations"). As written, a reviewer who opens the thresholds files will conclude the central methodological claim is unsupported.

### A3. CRITICAL — the "safeguard" consumes ground truth at test time; deployability is overclaimed

> "these results suggest a clinically generalizable pathway for improving the reliability of LLM-based decision support through post-hoc safeguards, without requiring task-specific retraining"

At **test** time, the pipeline passes the dataset's reference answer (`truth`) into all three intervention components (see C1): the claim validity scorer ("If the claim contradicts the Ground Truth … score much LOWER"), the risk scorer, and the rewrite prompt ("Reference Evidence (use this as ground truth if provided)"). The measured reductions are therefore for an **oracle-assisted** system that cannot exist at deployment, where no reference answer is available. "Clinically generalizable pathway" and "post-hoc safeguards" strongly imply deployability. The abstract nowhere discloses the oracle.

**Fix:** Disclose ground-truth access explicitly (e.g., "assuming access to reference evidence, as in retrieval-augmented settings") and temper "clinically generalizable pathway," or re-run with `truth=None` at test time and report those numbers.

### A4. HIGH — "average reductions of up to 7.4%" is ambiguous, cherry-picked, and hides the tail

> "achieving average reductions of up to 7.4%"

- Traceable to the HealthSearchQA macro-average of overall sycophancy across the 6 models: 51.8% → 44.5% = **7.35 percentage points** (rounds to 7.3, not 7.4 — check the exact aggregation used; per-mode averages give −12.4pp in-context / −2.3pp preemptive, so no obvious statistic yields exactly 7.4).
- Units: this is percentage **points**, not percent. "7.4%" reads as either; the relative reduction is ~14%.
- "up to" means "on the better of the two datasets." MedQuAD's average reduction is only ~1.6pp, which the abstract never states — so the one quantitative claim in the abstract is the best-case number.
- The average conceals that one model worsens by 17.8pp (A1), and the "overall" metric counts *progressive* (error-correcting) flips as sycophancy, so part of the "reduction" is CCR suppressing beneficial corrections (e.g., Llama-3B HealthSearchQA progressive drops 7.0%→2.2% while regressive explodes).
- Comparability caveat: baseline rates are computed over all 1000 items (8000 rebuttal steps) while CCR rates cover only the 750-item test split (6000 steps), and baselines are judged on raw answers while CCR outputs are judged on claim-reconstructed text. The comparison is not item-paired (see C15).

**Fix:** Report both datasets with units: "average reductions of 7.3 percentage points on HealthSearchQA and 1.6 on MedQuAD," computed on the shared test split, and state the per-model range including the negative case.

### A5. MEDIUM — treating reductions in *progressive* sycophancy as protective

> "We further analyze different forms of sycophancy, including regressive and progressive variants, and find that CCR lowers sycophancy in most settings"

Progressive sycophancy (incorrect → correct after pushback) improves answer accuracy; only regressive sycophancy is the safety risk the opening sentences describe. Lumping both under "CCR lowers sycophancy" frames the suppression of beneficial corrections as a win. In several runs CCR sharply cuts progressive flips (Llama-3B HealthSearchQA 7.0%→2.2%) or inflates them (Gemma-4B MedQuAD 9.0%→24.5%) — these have opposite clinical value but move "overall sycophancy" the same way.

**Fix:** Make regressive sycophancy the primary endpoint (regressive macro-averages: MedQuAD 25.5%→22.6%, HealthSearchQA 41.8%→35.8%) and report progressive separately as a cost/benefit, not as part of a single rate to be minimized.

### A6. MEDIUM — dataset naming and provenance

> "two widely used healthcare question-answering datasets, MedQuad and HealthSearchQA"

- Standard spelling is **MedQuAD** (also misspelled throughout the repo as `medquad`/`MedQuad`; at minimum fix the paper).
- The official HealthSearchQA release (Med-PaLM) is a set of consumer health *questions* — it ships no free-form reference answers. The loader (`data_loader.py:31-51`) consumes a `healthsearch_qa.jsonl` with `Free_form_answer`/`Must_have`/`Nice_to_have` fields whose provenance is documented nowhere in the repo. If these answers were model-generated, the entire HealthSearchQA ground truth (which also drives the judge, the claim scorer, and the rewrite oracle) is synthetic, and the paper must say so.

**Fix:** Correct the name; add a data-provenance statement for the HealthSearchQA answer key.

### A7. LOW — writing quality

- "investigate conformal risk control as a protective mechanism against it" — "it" dangles two clauses back from "sycophancy"; tighten.
- The CCR definition sentence (~45 words) chains four stages plus a purpose clause; split it. It also misorders the mechanism: reconstruction happens by concatenating kept claims, and the rewrite is a *separate*, response-level intervention triggered by a risk score — "before reconstructing the final response through a rewrite intervention" describes neither accurately.
- "evaluate sycophantic behavior across multiple model families and parameter scales" — the v6 evidence is three families at 1B–4B; consider stating the range so "parameter scales" isn't read as including frontier-scale models.

---

## Part 2 — Code issues

### C1. CRITICAL — ground truth is injected into the test-time intervention (silent oracle)

- `conformal_v2/run_conformal_v2.py:560` and `:598` — both test-phase `purify_answer_with_claims(...)` calls pass `truth=truth`, so the claim scorer's prompt (`scorers.py:48,59`: "If the claim contradicts the Ground Truth … score much LOWER") filters claims against the reference answer at test time. The parameter's own comment (`run_conformal_v2.py:87`) says "Ground truth allowed for calibration" — the test path violates it.
- `conformal_v2/run_conformal_v2.py:585` — the rewrite call passes `truth=truth`; `safe_rewrite.py:107` puts it in the prompt as "Reference Evidence (use this as ground truth if provided)". This directly contradicts the module's design goals: `safe_rewrite.py:18` — "2) Do NOT use ground truth: no access to the dataset reference answer."
- `conformal_v2/run_conformal_v2.py:549` — the test-time risk scorer also receives `truth` (`syco_risk.py:89` inserts a "Ground Truth (Correct Answer)" block and instructs low risk "if the change was toward the Ground Truth").

**Impact:** every reported CCR number measures an oracle-assisted system; the headline reductions are not achievable in deployment.
**Fix:** In the test path, call all three with `truth=None` (keep truth only in `judge_local` and `judge_claim_support`, which are legitimately evaluation-side); re-run. If oracle access is intentional (retrieval-style assumption), document it prominently in code and paper, and update the `safe_rewrite.py` docstring.

### C2. CRITICAL — calibration silently degenerates; runs proceed and results get reported

- `conformal_v2/conformal_thresholds.py:145-146` — `fit_global_threshold` returns the sentinel `-1.0` when no threshold satisfies α. In `fit_thresholds` / `main` this value is used as tau **without any warning**: every `s ∈ [0,1] > −1.0`, so `--enable_rewrite` rewrites 100% of drafts. The saved thresholds confirm this happened in 10 of the 12 v6 runs (`results/*_v6_1000/thresholds_*.json`: `"tau_global": -1.0`).
- `conformal_v2/run_conformal_v2.py:374-384` — the claim-level fallback (`tau_claim = 0.3`) at least logs a warning, but the run continues and the output metadata records `"claim_threshold": 0.3` indistinguishably from a genuinely calibrated 0.3. All 12 v6 thresholds files contain `tau_claim = 0.3`.

**Impact:** the "conformal" experiment can silently degrade to "always rewrite + fixed claim filter" and still produce plausible-looking result files; that is exactly what the committed results are.
**Fix:** On `-1.0`, log loudly, write an explicit `"calibration_failed": true` flag into the thresholds/output metadata, and decide the policy consciously (abort, or run an explicit "always-rewrite" arm). Same for the claim fallback: persist `"tau_claim_fallback": true`. Report `rewrite_triggered` rates (already stored per row) in the summary block so degeneration is visible without opening per-row records.

### C3. CRITICAL — risk score is computed on different inputs in calibration vs. test

- Calibration: `conformal_v2/run_conformal_v2.py:408-417` scores `draft_answer=purified_answer` (after claim filtering).
- Test: `:543-552` deliberately scores the **raw** draft (comment: "Score risk on the RAW draft BEFORE purification").

The threshold tau is fit on the distribution of scores of purified drafts and applied to scores of raw drafts. Conformal/risk-control validity requires the identical score function on calibration and test; with different inputs there is no exchangeability and the α-guarantee is void even where calibration "succeeds" (the two Phi runs). The calibration "bad" label (`:397-406`, regressive on the *purified* draft) likewise doesn't match the test-time decision object.
**Fix:** score the raw draft in calibration too (matching the test comment's rationale), and define `bad` on the same object the threshold gates.

### C4. HIGH — `--use_group_thresholds` is accepted, recorded, and ignored

- `conformal_v2/run_conformal_v2.py:763` and `:895` hardcode `use_group_thresholds=False` in both the `calibrate` and `both` paths, while metadata records the CLI flag's value (`:782`, `:946`).
- All v6 SLURM jobs pass `--use_group_thresholds` (e.g. `slurm/v6_llama_1b_conformal.slurm`), and the suite header advertises "Group-Thresholds" methodology — but every thresholds file shows `"tau_by_group": null`. The recorded experiment config claims a methodology that never executed.
- Latent secondary bug: even if enabled, group thresholds wouldn't survive a save/load cycle — `threshold_to_json` (`:655`) stringifies tuple keys, `thresholds_from_json` (`:683`) keeps them as strings, but `choose_threshold` is called with tuple keys (`:571-573`), so lookups would silently fall back to `tau_global`.

**Fix:** pass `use_group_thresholds=bool(args.use_group_thresholds)` at both call sites; serialize group keys reversibly (e.g., JSON list keys or a parallel list); add a test that a fitted group threshold round-trips and is actually selected.

### C5. HIGH — risk-score parsing grabs the first number in the *analysis*, not the "Final Score"

`conformal_v2/syco_risk.py:24` (`_FLOAT_0_1_RE`) + `:41` (`.search` over the whole output). The prompt (`:115-118`) instructs "Provide analysis first … Final Score: <number>", so the model's free-text analysis precedes the score — any bare `0`, `1`, `0.5`, "step 1", "option 1" in the analysis is parsed as the risk score. Compounding it:

- `models.py:246` — `max_new_tokens=256` (not overridable via `ask_model`) can truncate the output *before* "Final Score:" is emitted; the parser then returns the max-risk default `1.0` (`syco_risk.py:43-44`), and parse failures are not logged/counted.
- With tau = 0.9 (the two Phi runs) this decides rewrites; in calibration it distorts the score distribution that tau is fit on for every run.

**Fix:** extract the score from the `Final Score:` line specifically (regex anchored on the label, fall back to last-number-in-text), raise the token budget for the scorer call, and log/count parse failures in the output metadata.

### C6. HIGH — risk scorer is non-deterministic and ignores both the config temperature and its own parameter

`conformal_v2/syco_risk.py:120-127`: `ask_model(..., temperature=0.7, ...)` is hardcoded; the function's `temperature` parameter (and the run-level `--temperature 0.0`) are silently ignored, while run metadata records `"temperature": 0.0`. No random seed is ever set for HF sampling (`models.py` has no `torch.manual_seed`; `do_sample=True` at `models.py:274-276`). Consequences: risk scores — and therefore rewrite decisions and fitted taus — are irreproducible run-to-run, contradicting `safe_rewrite.py:76` ("keep deterministic for research reproducibility") and the recorded config.
**Fix:** honor the passed temperature (make 0.7 an explicit CLI arg if desired), seed torch generation, and record the *effective* scorer temperature in metadata.

### C7. HIGH — `mode=calibrate` → `mode=test` workflow tests on the calibration data

`conformal_v2/run_conformal_v2.py:741-745` loads `all_data` identically in every mode (same default `--seed 7`, same `--max_items`), and `mode=test` (`:806-813`) evaluates on **all** loaded items. The documented two-step workflow (module docstring, options 2/3) therefore evaluates on the exact items used for calibration — full leakage — with no error, and the thresholds file records nothing about which items produced it. Only the single-run `mode=both` path splits (`:875-879`). The v6 runs used `both`, so the committed results are safe from this specific leak, but anyone following the docstring workflow is not.
**Fix:** persist the item ids (or seed + split indices) in the thresholds JSON and have `mode=test` exclude them (or at least hard-warn on overlap).

### C8. HIGH — stray private-API import can break the whole pipeline

`metrics.py:7` — `from pandas.core.missing import F`. `F` is a private typing alias, unused in the file (clearly an IDE auto-import). `metrics` is imported by `run_eval.py` and `run_conformal_v2.py`, so on any pandas version where that symbol moves, every entry point dies with `ImportError` before doing anything. Combined with the unpinned `requirements.txt` (`pandas` unversioned), this is a likely first-run failure for anyone reproducing.
**Fix:** delete the line.

### C9. HIGH — unresolved git merge conflict committed in `evaluation_analysis.py`

`evaluation_analysis.py:10` (`<<<<<<< Updated upstream`) and `:28` (`=======`) — the file contains raw conflict markers and is not valid Python; it cannot be imported or run. The conflicted half also hardcodes another user's absolute path (`/Users/lucianom/Research_realiability_LLMs/...`, `:29`).
**Fix:** resolve the conflict (keep the multi-file loop variant), remove the absolute path.

### C10. MEDIUM — one unparseable judge reply crashes a 7-day run; results are written only at the end

`judge.py:55` raises `ValueError` when the judge's output contains none of the three labels. Neither `calibration_collect`, `test_apply`, nor `run_medquad` wraps judge calls in try/except, and both runners write their output JSON only after the full loop (`run_conformal_v2.py:969`, `run_eval.py:270`). A single malformed judge response (or transient model glitch) at hour 100 of a `gpu_7day` job discards everything.
**Fix:** catch per-item, retry once, then record label `"erroneous"` + a `judge_parse_error` flag; checkpoint rows incrementally (append JSONL) instead of one final dump.

### C11. MEDIUM — claim text is mangled by `lstrip` (leading numbers/doses eaten)

`claims.py:20` — `line.strip().lstrip("- ").lstrip("1234567890. ")` strips *any* of those characters repeatedly, so claims that legitimately start with digits are corrupted: "500 mg daily is the maximum dose" → "mg daily is the maximum dose"; "2019 guidelines recommend…" → "guidelines recommend…". In a medical-claims pipeline the numeric lead of a claim is often the claim. Corrupted claims then flow into scoring, filtering, reconstruction, and the correctness judge.
**Fix:** strip list markers with a pattern, e.g. `re.sub(r"^\s*(?:[-*•]\s*|\d{1,3}[.)]\s+)?", "", line)`, which removes "1. " / "- " but not "500 mg".

### C12. MEDIUM — accepted drafts are purified twice in the test path

`conformal_v2/run_conformal_v2.py:554-560` purifies the draft (rebinding `draft_answer` to the purified text — the raw draft is never stored in the row, though line `:619`'s field name suggests it is), then `:590` sets `final_raw = draft_answer` for accepted drafts and `:592-598` purifies **again**. The second pass re-decomposes and re-scores an already claim-shaped string: it doubles the LLM calls per accepted draft, can drop additional borderline claims (so the judged "final" differs from what calibration modeled), and the rewrite branch (`:579-588`) receives the *purified* draft while the risk score was computed on the *raw* one.
**Fix:** purify once; if `rewrite_triggered`, purify only the rewrite output; store both raw and purified draft texts in the row.

### C13. MEDIUM — baseline and CCR are compared on different item sets and different judged artifacts

- `slurm/v6_*_baseline.slurm` runs `run_eval.py --max_items 1000` (all items), while the conformal runs test on the last 750 items of the same sample (`calib_frac 0.25`). `fix_table3.py:29-30` then compares the two directly, so the "baseline" includes 250 items the CCR arm never tested on, and the comparison is unpaired.
- The baseline judges raw model answers; CCR judges claim-reconstructed concatenations (`reconstruct_answer` = space-joined claims) — plus the literal placeholder string "(No valid claims found)" when everything is dropped (`run_conformal_v2.py:77`), which the judge will grade (typically erroneous → counted as regressive if the first answer was correct). Part of the measured deltas may be judge reaction to text style, not sycophancy.

**Fix:** restrict baseline aggregation to the test-split indices (they're recoverable: same loader, seed 7, order-preserving split) and add a "baseline + purification, no rewrite" arm to isolate the rewrite effect from the text-style effect.

### C14. MEDIUM — threshold selection invalidates its own confidence bound; claim calibration double-dips

- `conformal_v2/conformal_thresholds.py:119-148` scans every candidate tau and picks the largest whose *95%* Wilson upper bound ≤ α. Selecting the max over many data-dependent candidates is multiple testing; the post-selection guarantee is weaker than the nominal 95% (RCPS-style approaches handle this with fixed-sequence testing or monotonicity assumptions). The hidden confidence level (`z=1.96`, `:46`) is also never surfaced as a (α, δ) pair.
- `conformal_v2/run_conformal_v2.py:372-396`: `tau_claim` is fit on the calibration claims and then used to purify the *same* drafts from which the response-level `(s, bad)` pairs are computed — nested reuse of the same data, so even the response-level calibration is not clean split-conformal.

**Fix:** document the (α, δ) semantics; use a fixed-sequence scan (from most conservative tau) or Learn-then-Test correction; if feasible, use disjoint calibration halves for the claim and response levels.

### C15. MEDIUM — `summarize_rates` denominators conflate opportunity with behavior

`metrics.py:44-53` divides progressive, regressive, and overall counts by **all** rebuttal steps. Only initially-correct steps can be regressive and only initially-incorrect steps progressive, so these rates are confounded by each model's initial accuracy (e.g., Phi-2's low "regressive rate" partly reflects that it is initially wrong more often — fewer opportunities). Cross-model and cross-paper comparisons (SycEval conditions on eligibility) are distorted, and "overall" additionally counts beneficial progressive flips as sycophancy.
**Fix:** report regressive rate conditioned on `first_label == "correct"` and progressive conditioned on incorrect/erroneous; keep the unconditional rates only as secondary.

### C16. MEDIUM — progressive-direction rebuttals are generated blind to the reference answer

`rebuttals.py:22-27` — in the `flip=False` branch (initial answer judged incorrect), the prompt asks the 7B rebuttal model to "write the best concise correct answer" **without including `target_answer`** (the parameter is unused in that branch). The "correct" claim pushing the model toward truth is whatever the rebuttal model believes, which may itself be wrong — mislabeling some progressive-pressure trials. **Fix:** include `target_answer` (the dataset truth) in the flip=False prompt.

### C17. LOW-MEDIUM — `find_case_example.py` reads the thresholds file with wrong keys and a broken sentinel check

- `find_case_example.py:41-42` — `thresholds.get("global", {}).get("tau", -1.0)` and `thresholds.get("claim_threshold", 0.3)`; the actual schema (`threshold_to_json`) is `tau_global` / `tau_claim`. Both lookups always miss and silently use the defaults — which only coincidentally equal the saved values (−1.0 / 0.3). If real thresholds are ever calibrated, the case-study script will ignore them.
- `:135` — `if final_answer == "KEEP_DRAFT"`: `anti_sycophancy_rewrite` returns the text after "Final answer:", never the bare decision token, so this branch is effectively dead.
- Default `--results_file` is a distill-experiment file whose `qid`s index a different sampling than `load_data_local(n=1000, domain="healthsearch")` at `:36` — candidate indices may point at the wrong questions.

**Fix:** read `tau_global`/`tau_claim`; parse the `Decision:` line explicitly; assert the results file's domain/sample matches the loaded data.

### C18. LOW-MEDIUM — duplicated, diverged, shadowed package `SycEval_implementation/`

The inner `SycEval_implementation/` directory duplicates `scorers.py`, `conformal.py`, `conformal_wrapper.py`, `claims.py`. Its `scorers.py` is an older, different implementation (no rebuttal/truth conditioning, different parse fallbacks) — but it is dead code: the inner `conformal_wrapper.py` does `from scorers import ...`, which resolves to the **root** modules at runtime. `run_eval.py:13` imports through the inner package name, which only works with CWD = repo root (namespace package). Two same-named `score_claim_sycophancy` with different semantics is a trap for reproducers.
**Fix:** delete the inner directory and import `conformal_wrapper` from the root (or make the repo a proper installable package).

### C19. LOW — `metrics.py` micro-bugs

- `metrics.py:80` — `katz_log_rr_ci` returns `(rr, rr, hi)`; the computed lower bound `lo` (`:78`) is discarded, so any consumer prints the point estimate as the CI lower bound.
- `two_proportion_z` (`:31-34`): fine, but note `p1`/`p2` are passed as proportions and multiplied back by n — OK, just brittle if anyone passes counts.

**Fix:** `return (float(rr), float(lo), float(hi))`.

### C20. LOW — `statistical_significance.py` pairing can explode

`statistical_significance.py:133-137` merges teacher/student on `["qid", "run_id"]` only; with multiple rows per qid (2 modes × 4 strengths), the merge is many-to-many (up to 64 pairs per qid instead of 8), inflating McNemar's N and mispairing conditions. The inline comment even acknowledges it ("maybe add mode/strength if needed for uniqueness"). Affects the distillation analysis, not the CCR tables.
**Fix:** merge on `["qid", "run_id", "mode", "strength"]` and assert 1:1.

### C21. Reproducibility blockers (grab-bag, all LOW individually, HIGH collectively)

1. **HealthSearchQA data file is almost certainly untracked**: `.gitignore:209-213` ignores `*.jsonl` (and `*.json`) globally; `data/healthsearch_qa.jsonl` exists locally but a fresh clone won't have it, and no script recreates it. (`results/*.json` are tracked despite the ignore — presumably force-added — which is itself confusing.) → Verify with `git ls-files data/`; commit the file or add a download/provenance script.
2. **No documentation for the CCR pipeline.** `README.md` documents only the Ollama baseline (`run_eval.py`) with different models than any reported experiment; `RUNNING.md` hardcodes another machine's path (`/Users/lucianom/...`, line 9) and describes an outdated `load_data_local(csv_path=...)` signature. Nothing explains `conformal_v2.run_conformal_v2`, the v5-vs-v6 suites, or how to regenerate the tables (`fix_table3.py` is undocumented). The repo contains three overlapping pipelines (`run_conformal.py` v1, `run_eval.py --enable_conformal` wrapper, `conformal_v2/`) with no statement of which produced the paper's numbers.
3. **Unpinned dependencies**: `requirements.txt` pins nothing except `transformers>=4.46.0`; `statsmodels` (needed by `statistical_significance.py`) is missing entirely.
4. **Seeds**: `--seed` (default 7) is not recorded in the conformal output metadata; `run_eval.py` has no `--seed` flag at all (silently always 7) — if either side ever changes it, baseline/CCR item sets silently diverge. No torch seed (see C6).
5. **Hardcoded local model dirs** (`models/Llama-3.2-1B-Instruct`, …) in all SLURM scripts, with a download script covering only Qwen/Mistral.
6. **Misleading SLURM logs**: v6 job scripts echo "V5 Conformal SKEPTIC" / "N=1000, MedQuad Only" while running v6 on HealthSearchQA; baseline outputs are named `*_v5.json` inside `*_v6_1000/` directories. Anyone auditing logs/filenames will mispair suites.
7. **Judge = rebuttal generator = risk scorer** (Qwen2.5-7B-Instruct for all three, `risk_scorer_model` defaulting to the judge at `run_conformal_v2.py:736`): the model that writes the misleading rebuttals also grades correctness and risk. Cheap to note in the paper; ideally use a disjoint judge.
8. `Optional` is referenced but never imported in `conformal_v2/syco_risk.py:71` and `conformal_v2/safe_rewrite.py:74` — currently harmless only because of `from __future__ import annotations`; breaks under runtime type inspection and any strict linter.
9. `.DS_Store` is modified/tracked; add it to `.gitignore`.

---

## Suggested triage order

1. C1 (remove/disclose the test-time oracle) and C2/C3 (make the conformal layer real or reframe it) — these determine what the paper can claim.
2. A1–A4 — rewrite the abstract's claims to match whichever system the final numbers describe.
3. C4–C7 — flags/scoring/leakage fixes needed before any rerun.
4. C8–C13 — correctness/robustness fixes worth doing in the same pass.
5. C21 — packaging/docs so the rerun is reproducible by someone other than the author.
