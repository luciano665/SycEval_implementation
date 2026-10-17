# Prompt: second-pass, in-depth review of `sigconf.tex` (paste into a fresh Claude Code session)

*Saved 2026-10-17. Pass 1 is `docs/PAPER_REVIEW.md`. Run this in a **new** session (not `--continue`) so the reviewer reads the files cold and can genuinely disagree with pass 1.*

**Write policy (agreed):** the session may create and commit **new files under `docs/` only**, never push. Rationale: HPC scripts must travel by `git pull` — pasting multi-line code into the cluster terminal drops newlines — while no-push keeps the shared remote under the user's control. The user pushes after `git show --stat`.

---

```
Before you start, check whatever skills, superpowers, or review tools you have access to and use any that fit (code-review, systematic-debugging, verification-before-completion). If none fits a step, proceed carefully and say so.

CONTEXT
Repo: SycEval_implementation, branch conformal_v10. Paper: sigconf.tex (ACM sigconf, "Auditing Sycophancy with Conformal Risk Control"). A first-pass review already exists at docs/PAPER_REVIEW.md with raw HPC output in docs/PAPER_REVIEW_hpc_output.txt, and a verification script hpc_verify_paper.py. That pass checked that reported numbers match result files and found 4 critical issues (C1–C4), which have since been fixed in sigconf.tex (commit 24c6bec). Read docs/PAPER_REVIEW.md first so you don't repeat it — your job is the second, deeper pass. Treat its findings as claims to re-verify, not as facts.

SAFETY RULES — NON-NEGOTIABLE
- READ-ONLY on everything except new files under docs/ (your report and helper scripts) and your scratchpad. Do not edit sigconf.tex, any .py outside docs/, any .bib, .gitignore, or anything under results/ or data/.
- You may create new files only under docs/. You may `git add` and `git commit` those new files only — never modified or deleted files, never anything outside docs/ — and only after printing `git status` and `git diff --cached --stat` and confirming every listed path is a new file under docs/. Never `git push`; I will push myself. No other git commands that change state (checkout, reset, stash, rebase, rm, branch).
- No rm, mv, or overwrite of any existing file. If you believe a change is needed, describe it in the report; do not make it.
- Do not run any experiment, model, or SLURM job. Do not call any LLM API. If a check needs the HPC or a model, write the exact command for me to run and mark the item HPC-PENDING.
- Before any command that writes outside your scratchpad or docs/, stop and ask.
- Evidence standard: every claim you make about the paper, code, or data must cite path:line (or a URL for external sources). If you cannot find a source, write UNVERIFIED — never assume correctness, and never infer a number from a tracker or markdown note when a raw file exists. Trackers under docs/ are secondary evidence only.
- Distinguish clearly between "I verified this", "this is internally consistent but unverified", and "this is my judgment". Flag your own uncertainty explicitly.

SCOPE — GO DEEPER THAN PASS 1 IN THESE TEN AREAS

1. Pipeline correctness audit (conformal_v2/run_conformal_v2.py, conformal_v2/conformal_thresholds.py, conformal_v2/syco_risk.py, conformal_v2/safe_rewrite.py, scorers.py, claims.py, judge.py, rebuttals.py, metrics.py, models.py, data_loader.py, run_eval.py). Read these files in full, not by grep. Look for bugs or design choices that would invalidate or bias a reported result:
   - Any path by which reference answers ("truth") reach the claim scorer, risk scorer, or rewrite at test time. Trace every call site and every default argument.
   - Calibration/test leakage: the 200/800 split, the question-hash pairing (_question_hash, exclude_calibration_items), duplicate question texts, and whether any test-time quantity was fit on test data.
   - The claim-threshold inversion (claim_scores = 1 - v, tau_claim = 1 - tau_inverted): prove it is direction-correct, and check edge cases (ties, v exactly equal to tau, empty claim lists, parse failures).
   - The exact-CRC threshold scan (fit_global_threshold_exact_crc, certifiable_fraction): tie-breaking, the ">= best_accept_count" rule, whether alpha_min as computed equals min over tau of (k+1)/(n+1) exactly, and whether the implementation matches eq. (crc) and eq. (alpha_min) in the paper.
   - Wilson implementation (z, continuity, clamping) vs. the paper's "95% Wilson upper bound".
   - Parse-failure handling: risk score -> 1.0, judge -> "erroneous", claim scorer fallbacks. Quantify from the raw records (HPC-PENDING command if needed) how often each fires, and whether that materially affects any reported rate.
   - Judge prompt and rebuttal prompts: anything that could bias labels toward the paper's conclusions (e.g., the judge seeing the rebuttal, or the "citation" strength using a template that models might recognize as fake).
   - Decoding determinism claims: temperature 0, batch size 1, HF vs ollama backends — confirm from models.py that the final runs' backend matches the paper.
   - Dead or unused code paths that the paper nevertheless describes (e.g., group thresholds, oracle flags).

2. Independent re-derivation of every table. Do NOT reuse analyze_*.py or hpc_verify_paper.py. Write your own script from the record schema (docs/PAPER_REVIEW_hpc_output.txt section E lists the keys) that recomputes Tables 1–5 and every in-text statistic from run_baseline_*.json / run_conformal_*.json / thresholds_*.json, including denominators and how "erroneous" is treated. Since the per-item JSONs are HPC-only, write the script, test it against the 12 committed thresholds files for the parts you can, commit it under docs/, and tell me to `git pull` and run it on the HPC.

3. Statistical rigor. The paper reports no CIs. Write a script (same delivery rules) that computes question-level clustered bootstrap 95% CIs (resample questions, not drafts; 8 correlated drafts per question) for every Table 1 rate, every Table 3 delta, and every Table 4 delta, plus paired differences between datasets and between model sizes within family. Then list every sentence in sigconf.tex that asserts a difference, ordering, or "larger/smaller" and classify each as: supported outside the CI, inside the CI (should be hedged), or untestable. Pay particular attention to: "higher on HealthSearchQA for all six", "larger model more susceptible in every family", "within about six points", "no model's effect changes sign", and the Phi-1.5 HSQA certification at alpha_min = 0.081.

4. Citation content audit. For every \cite in sigconf.tex, check that the cited work actually supports the specific claim attached to it. Fetch the abstract/paper where possible. Explicitly verify: FlipFlop "46% of the time"; Chen et al. "up to 100%"; Perez "grows with model scale"; Wei "scaling and instruction tuning increase it"; Sharma on preference models; Mohri & Hashimoto's claim-filtering design; Angelopoulos & Bates CRC bound form (k+1)/(n+1) for B=1; Learn-then-Test; Cherian's remark about discarding accurate claims; Singhal's HealthSearchQA having no answers; OLAPH/MedLFQA generating the answers; MedQuAD 47,457; yang2024 being the Qwen2.5 report. Report any attribution that is wrong, overstated, or that cites a different version (arXiv vs. venue) than the .bib claims. Note: the .bib is not in the repo — list every key and mark its bibliographic details UNVERIFIED unless you can confirm them from the paper's own text.

5. LaTeX and submission compliance. If a TeX toolchain is available (check pdflatex/latexmk/tectonic; try a Docker TeX Live image only if docker is present and you ask first), compile sigconf.tex with a stub .bib containing all 31 keys and report every error and warning. If no toolchain, statically check: acmart option compatibility (sigconf, review, anonymous), required packages (\checkmark, \mathbb, \sout, \multirow, tcolorbox, booktabs), float placement, the CCS block requirement, overfull-risk long table rows, and anonymity leaks (author names, institution, grant numbers, file paths, GitHub URLs, the GenAI disclosure, acknowledgments, metadata in \acmDOI/\acmISBN). Also check that every \label is referenced or justified, every table/figure is referenced before it appears, and that eq. labels used in prose exist.

6. Reproducibility. Confirm from code + thresholds files that the exact N=1,000 sample can be regenerated from seed 7 (data_loader.py sampling) and that the 200 calib_question_hashes in each thresholds file are identical across the six models of a dataset and consistent with the first 200 of that sample. Verify every SLURM arg in slurm/final1k*.slurm against the paper's Configuration section. Check that the 12 committed thresholds files are byte-consistent with the HPC originals (give me a sha256sum command).

7. Internal consistency sweep. Build a table of every number that appears more than once (abstract, intro, results, discussion, conclusion, captions, limitations) and confirm each instance agrees. Check qualifiers too: "all", "every", "none", "only", "first", "no model", "both datasets" — each must be literally true given the tables.

8. Limitations and overclaim check. List claims a skeptical reviewer would challenge and whether the paper pre-empts them: single-judge confound, model-generated HSQA references, no selective intervention ever exercised, ≤4B models, fabricated-citation rebuttals, 0.3 fallback chosen how and why, alpha_min selection bias, exchangeability violation from 8 drafts/question, generated-rebuttal realism, the second claim-filter pass on rewrites (now disclosed), and anything else you find. Say which are adequately addressed, under-addressed, or missing.

9. Data and licensing. Confirm licenses/terms for MedQuAD (NIH sources; MedlinePlus removals), the keivalya HF derivative, HealthSearchQA, OLAPH/MedLFQA, and all six model weights plus Qwen2.5-7B-Instruct, and whether the Ethics statement's "public datasets, no PHI" and the model usage are compliant. Flag anything that needs a license mention or a data-availability statement.

10. Re-audit of pass 1. Independently re-verify every C and S item in docs/PAPER_REVIEW.md and the fixes applied in commit 24c6bec. If you disagree with any finding, or find the applied fix introduced a new error (wrong number, broken sentence, bad LaTeX), say so explicitly in a "Disagreements with pass 1" section. Be adversarial toward the earlier report, not deferential.

DELIVERABLES
A. docs/PAPER_REVIEW_V2.md containing:
   - Executive summary: what you checked, counts of verified / unverified / HPC-pending / wrong, and the top issues ranked by how much they would hurt at review.
   - Findings in three tiers — Critical (invalidates or materially biases a result, or a factual/attribution error), Should-fix (rigor, clarity, compliance), Minor — each with: exact quoted .tex text and line, the evidence (path:line or URL), what is actually true, and a concrete suggested fix (as text to paste; do not apply it).
   - A "Disagreements with pass 1" section.
   - A "Scripts for the HPC" section with every HPC-PENDING command, each stating exactly what number it confirms and what value to expect.
   - A full claim ledger (one row per claim, status, evidence).
B. The same executive summary in your reply, plus a completeness score /100 with a one-line justification per deducted point.
C. End your reply with the exact `git push` command and a one-line list of the files you committed, so I can review and push.

Do not edit sigconf.tex or any code. Do not push. If at any point you are unsure whether an action is read-only, ask before doing it.
```
