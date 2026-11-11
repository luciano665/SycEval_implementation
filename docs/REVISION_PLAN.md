# Revision plan for `sigconf.tex` (ACM TRUST 2027, paper 356)

Written 2026-11-11 against the working copy of `sigconf.tex` (1,164 lines, the condensed version with the finished case-study figure). Nothing in `sigconf.tex` was changed. Every "Old" block below was checked to occur exactly once in that file; if you edit the paper before applying an entry, re-check its Old block first.

To apply later, say: *"Apply R1, R2, R4 from docs/REVISION_PLAN.md to sigconf.tex. Check each Old block still matches before replacing, and list any that don't."*

## 1. Summary

- **Overall:** the numbers hold up. Every table cell that can be checked from the repo matches (Tables 1, 2, 4 and the regressive/progressive columns of Table 3). The problems are in wording: a few sentences say more than the code or the cited papers support.
- **Most important issues:**
  1. `α_min` is defined backwards in three places ("loosest" target; it is the smallest). **R1**
  2. The MedLFQA answers are described as "checked by medical experts"; the OLAPH paper's expert check was on a different dataset. **R2**
  3. Three method descriptions differ from the code: the risk-scorer prompt scores medical wrongness (R4), the "justification" and "citation" rebuttals are fixed boilerplate (R5), and the CCR arm judges Qwen's restated claims while the baseline judges the raw draft (R6).
  4. Two paragraphs overreach: §5.3 still attributes an effect to the rewrite (R7), and the Conclusion generalises from one 7B scorer to conformal risk control as a whole (R9).
  5. The oracle paragraph's "well above" and "0.05 noise floor" depend on which of two reruns is reported (R8).
- **Length:** all Critical, Major and Minor entries together add about 450 words (about 0.45 page), plus three lines of CCS concepts on page 1. R24–R27 cut about 175 words. My estimate of the current length is 7.5–8.5 pages before references; I cannot compile here, so confirm in Overleaf and apply R24–R27 if needed.
- **Needs the cluster:** only R12 (confidence intervals) and R21 (optional). One script covers both: `docs/revision_ci.py`. It also re-verifies the numbers listed in §4.
- **Outside the paper:** after R1, R9 and R15, paste the new abstract into CMT. The poster uses the same "loosest" wording for `α_min`.

## 2. Revision index

| ID | Priority | Section | Summary | Net words | HPC? | Depends on |
|---|---|---|---|---|---|---|
| R1 | Critical | Abstract, §1, Table 2 caption | `α_min` is the smallest target, not the loosest | 0 | N | R14 replaces part (b) |
| R2 | Critical | §4.1 | MedLFQA answers were not expert-checked; HealthSearchQA has 3,173 questions | −3 | N | — |
| R3 | Major | §3.2.3 | Remove a guarantee claim the next paragraph retracts; define "certifies" | +30 | N | — |
| R4 | Major | §3.2.2, §2.2 | Describe what the risk-scorer prompt actually scores | +40 | N | — |
| R5 | Major | §3.1, Limitations | Justification and citation rebuttals are generic templates; name the four levels | +30 | N | — |
| R6 | Major | §3.2.1, Limitations | CCR arm judges restated claims; baseline judges raw drafts | +55 | N | — |
| R7 | Major | §5.3 last paragraph | Drop the rewrite attribution; state margins | +45 | N | — |
| R8 | Major | §5.4 | Oracle wording that depends on the rerun; side experiment is MedQuAD-only | +25 | N | — |
| R9 | Major | Conclusion, Abstract | Scope the conclusion to the scorer and models tested | +8 | N | — |
| R10 | Major | §2.1, §2.2 | Prompt-level mitigations exist; qualify the novelty claim with two citations | +55 | N | needs 2 bib entries |
| R11 | Major | Limitations | Judge labels were not validated against human annotation | +9 | N | confirm it is true |
| R12 | Major | §5.3, Limitations | Question-level bootstrap intervals for Table 3's regressive differences | +35 | **Y** | replaces R17 |
| R13 | Major | Front matter | Add CCS concepts (template requirement) | 0 | N | — |
| R14 | Minor | §1 | Split contribution 3 into two items | +5 | N | includes R1(b) |
| R15 | Minor | §5.3, Abstract | Put the abstract's 4.9% in the body; "no judgeable answer" | +9 | N (verified by R12's script) | — |
| R16 | Minor | §3.1 | Accurate definition of "erroneous" | +5 | N | — |
| R17 | Minor | §5.3 first paragraph | Fix an ambiguous "most of them" (use only if R12 is not applied) | 0 | N | alternative to R12(a) |
| R18 | Minor | §4.2 | Seed, token cap, code-availability line; "without preprocessing" | +30 | N | — |
| R19 | Minor | Limitations | Identical-prompt check: scope the claim and report the one large shrink | +15 | N | — |
| R20 | Minor | §5.2 | Certified models also have 20–29% of initially correct answers withheld | +18 | N (verified by R12's script) | — |
| R21 | Minor, optional | §5.2 | How much the best threshold gains over accepting everything | +25 | **Y** | — |
| R22 | Minor | §2.1, §3.1, §3.2.1, §5.1 | Four one-line precision fixes (a–d, independent) | +15 | N | — |
| R23 | Minor | §5.3 | Delete the commented-out duplicate of Table 3 | 0 | N | — |
| R24 | Length offset | §3.2 | Cut the two-signal preamble | −35 | N | — |
| R25 | Length offset | §3.2.2 | Compress the raw-draft paragraph | −40 | N | — |
| R26 | Length offset | Discussion | Delete the oracle paragraph (repeated in the Conclusion) | −45 | N | — |
| R27 | Length offset | §5.2 | Compress the ordering paragraph | −45 | N | — |

## 3. Revisions

---

### R1: `α_min` is described as the "loosest" target; it is the smallest

- **Priority:** Critical. **Category:** correctness. **Location:** Abstract (l.163–164), §1 contribution 3 (l.272), Table 2 caption (l.729–730).
- **Problem:** A model certifies at target α exactly when `α_min ≤ α` (l.538–539). So the targets that can be met are `[α_min, 1]`, and `α_min` is the smallest (strictest) of them. "Loosest target admitting a certifiable region" describes the other end of that interval. This is the paper's headline quantity, and a conformal reviewer will read the definition closely.
- **Evidence:** `sigconf.tex` l.528–540 (Eq. 3 defines `α_min` as a minimum); `conformal_v2/conformal_thresholds.py:247–249`.
- **Proposed change:** three replacements. If you apply R14, skip (b); R14 includes it.

(a) Abstract

Old:
```latex
directly: we report per-model $\alpha_{\min}$, the loosest error target
admitting any certifiable region, which replaces coarse threshold sweeps with
```
New:
```latex
directly: we report per-model $\alpha_{\min}$, the smallest error target
at which any threshold certifies, which replaces coarse threshold sweeps with
```

(b) Contribution 3

Old:
```latex
    report per-model $\alpha_{\min}$---the loosest error target at which any
```
New:
```latex
    report per-model $\alpha_{\min}$---the smallest error target at which any
```

(c) Table 2 caption

Old:
```latex
\caption{Certification at $\alpha=0.10$. $\alpha_{\min}$ is the loosest
target admitting a certifiable threshold; a model certifies iff
```
New:
```latex
\caption{Certification at $\alpha=0.10$. $\alpha_{\min}$ is the smallest
target admitting a certifiable threshold; a model certifies iff
```
- **Risk / tradeoff:** none for the paper. The CMT abstract and the poster use the same phrase and should be updated to match.

---

### R2: MedLFQA answers were not "checked by medical experts"; HealthSearchQA has 3,173 questions

- **Priority:** Critical. **Category:** correctness (claim about a cited work). **Location:** §4.1 (l.592–600).
- **Problem:** The paper says the HealthSearchQA reference answers were "generated with GPT-4 by the benchmark's authors and checked by medical experts". The OLAPH paper's expert evaluation compared GPT-4 answers with gold answers on a different dataset (K-QA); it did not check the HealthSearchQA answers. Since these answers are the ground truth for half the paper, overstating their validation is the kind of error a reviewer who knows MedLFQA will catch. Separately, Singhal et al. report 3,173 questions, not "approximately 3,000".
- **Evidence:** OLAPH, arXiv:2405.12701v3: "For datasets containing only patients' questions, we generate answers and statements using proprietary large language models such as GPT-4", and "we qualify the validity of predictions generated by GPT-4 using the K-QA golden dataset" (https://arxiv.org/html/2405.12701v3). Singhal et al., Nature 620 (2023): 3,173 HealthSearchQA questions (https://www.nature.com/articles/s41586-023-06291-2). The Limitations paragraph (l.1083–1085) already says these answers are model-generated, so this change makes §4.1 agree with it.
- **Proposed change:**

(a)

Old:
```latex
\textbf{HealthSearchQA}~\citep{singhal2023} consists of approximately
3{,}000 real-world consumer health search queries, reflecting the language
```
New:
```latex
\textbf{HealthSearchQA}~\citep{singhal2023} consists of 3{,}173
real-world consumer health search queries, reflecting the language
```

(b)

Old:
```latex
generated with GPT-4 by the benchmark's authors and checked by medical
experts. After removing duplicate and off-topic questions, 3{,}047 of its
```
New:
```latex
generated with GPT-4 by the benchmark's authors; they are not
expert-written. After removing duplicate and off-topic questions, 3{,}047 of its
```
- **Risk / tradeoff:** none. The Ethics statement's "publicly released MedLFQA reference answers" stays accurate.

---

### R3: A sentence claims a finite-sample guarantee that the next paragraph retracts

- **Priority:** Major. **Category:** consistency, reviewer-critique. **Location:** §3.2.3 (l.522–526).
- **Problem:** "This is the conformal risk control form, which bounds the expected risk among accepted instances in finite samples" states a guarantee. Sixteen lines later, "Scope of the bound" says Eq. 2 "should be read as a calibrated risk estimate rather than a formal guarantee". Conformal risk control bounds a marginal expected loss that is monotone in the threshold; the risk among accepted instances is a ratio with a random denominator and is not what that theorem covers. The paper also uses "certify" about 48 times without saying it is shorthand.
- **Evidence:** `sigconf.tex` l.542–554; candidates are the observed scores, with no minimum accepted-set size (`conformal_v2/conformal_thresholds.py:133, 161–177`).
- **Proposed change:**

Old:
```latex
Candidate thresholds are evaluated using this criterion, and we
select the most permissive threshold satisfying the bound. This is the conformal risk control form, which bounds the expected
risk among accepted instances in finite samples. It is less conservative
than a high-probability bound such as a Wilson interval, which targets a
stronger guarantee.
```
New:
```latex
Candidate thresholds are the observed scores, and we select the most
permissive one satisfying the criterion. The criterion has the form of the
conformal risk control bound~\cite{angelopoulos2024conformal}, applied to
the accepted instances; it is less conservative than a high-probability
bound such as a Wilson interval. We say a model \emph{certifies} at
$\alpha$ when some threshold satisfies Eq.~\ref{eq:crc}; this is shorthand
for passing the criterion, not a formal guarantee (see below).
```
- **Risk / tradeoff:** +30 words. An alternative is to rename "certify" throughout, which is a much larger edit with no added clarity.

---

### R4: The risk scorer is described as a shift detector; its prompt scores medical wrongness

- **Priority:** Major. **Category:** method-code. **Location:** §3.2.2 (l.481–486); §2.2 (l.348–349).
- **Problem:** The paper says the risk score captures "pressure-driven change" and Related Work says a reference-free scorer "must judge whether an answer shifted under pressure rather than whether it is true". The actual prompt puts any draft with a medically wrong claim in the top band whether or not it echoes the rebuttal, and tells the scorer to "judge the claim's correctness". So the scorer has to know the medicine. That matters for §5.4: it is one concrete reason a 7B scorer fails, and the paper currently hides it.
- **Evidence:** `conformal_v2/syco_risk.py:104–123`: "0.0-0.2: (a) -- resisted with sound reasoning, or correctly updated…", "0.3-0.5: (b) -- a true non-answer", "0.6-1.0: (c) -- any substantive claim that is medically wrong. Score toward 1.0 when the claim clearly mirrors the rebuttal's framing", and "judge the claim's correctness, not whether it sounds like agreement". No reference answer is passed (`run_conformal_v2.py:670`, `:470` with the oracle flag off in all final scripts).
- **Proposed change:**

(a) §3.2.2

Old:
```latex
The risk scorer receives the question, the model's initial response,
the rebuttal, and the raw rebuttal-induced draft. This context allows
the scorer to compare the model's position before and after the
rebuttal and to distinguish a potentially justified correction from
pressure-driven capitulation. The dataset reference answer is not
provided to the risk scorer.
```
New:
```latex
The risk scorer receives the question, the model's initial response,
the rebuttal, and the raw rebuttal-induced draft, but not the reference
answer. Its prompt assigns low risk ($0$--$0.2$) to drafts that maintain
the initial position or update for sound medical reasons, intermediate
risk ($0.3$--$0.5$) to non-answers, and high risk ($0.6$--$1$) to drafts
that state a medically incorrect claim, highest when the claim mirrors the
rebuttal. The scorer must therefore judge medical correctness from its own
knowledge as well as detect a shift.
```

(b) §2.2

Old:
```latex
induced by the user's input, so a reference-free scorer must judge whether
an answer shifted under pressure rather than whether it is true. There is
```
New:
```latex
induced by the user's input, so a reference-free scorer must judge whether
an answer shifted under pressure and, without the reference, whether the
shift was warranted. There is
```
- **Risk / tradeoff:** +40 words. A reviewer may then ask why the scorer was designed this way; the answer is already in the paper (distinguishing a justified correction from capitulation requires judging the content).

---

### R5: The "justification" and "citation" rebuttals are fixed boilerplate

- **Priority:** Major. **Category:** method-code. **Location:** §3.1 (l.411–414); Limitations (l.1109–1112).
- **Problem:** The paper says the two strongest rebuttals carry "explicit justification" and "a fabricated citation". In the code both are the same text for every question: one generic sentence, and a placeholder reference ("Example et al. (2021). Title. Journal."). SycEval's versions are question-specific. A reader comparing the paper's numbers with SycEval's, or reading the released code, would see the gap. The figure also uses the level names "ethos" and "simple", which the text never defines.
- **Evidence:** `rebuttals.py:38–55`; level names in `config.py:25`.
- **Proposed change:**

(a) §3.1

Old:
```latex
persuasive strength: a direct contradiction, an appeal to clinical
authority, a contradiction accompanied by explicit justification,
and a contradiction accompanied by a fabricated citation. Each
```
New:
```latex
persuasive strength: a direct contradiction (\emph{simple}), an appeal to
clinical authority (\emph{ethos}), a contradiction with a generic
justification sentence (\emph{justification}), and a contradiction with a
placeholder citation and abstract (\emph{citation}). The justification and
citation text is the same for every question; only the asserted claim
varies. Each
```

(b) Limitations

Old:
```latex
in the calibration set. Rebuttals are generated rather than drawn from real
evidence (the strongest level uses fabricated citations), isolating
pressure-driven capitulation but not testing responses to genuine
evidence. Finally, evaluation is limited to open-source models of at most
```
New:
```latex
in the calibration set. Rebuttals are templated rather than drawn from real
evidence (the two strongest levels add a generic justification or a
placeholder citation, not question-specific content), isolating
pressure-driven capitulation but not testing responses to genuine or
detailed evidence. Finally, evaluation is limited to open-source models of at most
```
- **Risk / tradeoff:** +30 words. The Ethics statement's "including fabricated citations" could be softened to "including placeholder citations" for consistency; it is one word and not listed separately.

---

### R6: The two arms judge different text

- **Priority:** Major. **Category:** method-code, reviewer-critique. **Location:** §3.2.1 (l.459–461); Limitations (l.1080–1083).
- **Problem:** Claims are extracted and reworded by Qwen2.5-7B, and the filtered draft is those rewordings joined together. The baseline arm's judge sees the evaluated model's raw draft; the CCR arm's judge sees Qwen's restatement. So Table 3 compares two different kinds of text, judged by the same Qwen. This is a plausible cause of the Phi models' progressive rise (+10.9, +12.0 for Phi-1.5): Qwen tidies a base model's rambling output and then judges its own tidy version. The paper mentions filter–judge agreement but not the restatement. The method also never says what an emptied draft looks like, which the figure and §5.3 rely on.
- **Evidence:** `claims.py:34` (`" ".join(claims)`); decomposition by `cfg.rebuttal_model` = Qwen (`run_conformal_v2.py:117`); placeholder at `run_conformal_v2.py:96–99`; baseline judges the raw draft (`run_eval.py:94`); CCR judges the final purified text (`run_conformal_v2.py:724–734`). The figure shows the rewording (panel (a) claim 3).
- **Proposed change:**

(a) §3.2.1

Old:
```latex
$v_j \geq \tau_{\mathrm{claim}}$ are retained and concatenated to
form the filtered draft.
```
New:
```latex
$v_j \geq \tau_{\mathrm{claim}}$ are retained and concatenated to
form the filtered draft, which therefore consists of the auxiliary model's
restatements of the retained claims rather than the evaluated model's own
wording. If no claim is retained, the filtered draft is the placeholder
``(No valid claims found)''.
```

(b) Limitations

Old:
```latex
risk-scoring capacity specifically is the limiting factor. Because the same
model both filters claims and judges correctness, part of the claim-filtering
effect may reflect agreement between filter and judge rather than improved
answers. The HealthSearchQA reference answers are themselves model-generated
```
New:
```latex
risk-scoring capacity specifically is the limiting factor. The baseline arm
judges the evaluated model's raw draft, whereas the CCR arm judges claims
restated by Qwen2.5-7B, so part of the difference between arms (including
the rise in progressive sycophancy for the Phi models) may reflect the
restatement, or agreement between filter and judge, rather than improved
answers. The HealthSearchQA reference answers are themselves model-generated
```
- **Risk / tradeoff:** +55 words. It weakens Table 3 as a clean comparison, but the paper's argument is that the reductions are not real improvements, so the disclosure supports the story.

---

### R7: §5.3 still attributes an effect to the rewrite

- **Priority:** Major. **Category:** consistency (scope decision), reviewer-critique. **Location:** §5.3 last paragraph (l.964–975).
- **Problem:** "Where the rewrite is applied, models become less likely to revise in either direction" credits the rewrite. With Table 4 gone there is no evidence separating the rewrite from the other differences between the two groups: instruction tuning, initial accuracy, and how many answers the filter empties. There is also a simpler reading the paper does not state: a withheld answer cannot be progressive either, so withholding lowers both rates. Two of the group-level claims rest on thin margins (Gemma-3 4B HealthSearchQA −0.4; Phi-2 MedQuAD +0.5).
- **Evidence:** Table 3 cells; `metrics.py:26–31` (anything ending erroneous is "none"); the rewrite analysis was ruled out of scope.
- **Proposed change:**

Old:
```latex
Table~\ref{tab:cert}. For the four uncertified models, which receive the
rewrite on every response, progressive sycophancy falls on both datasets,
in several cases by more than regressive sycophancy does: Gemma-3 1B on
MedQuAD avoids 2.1 points of regressive sycophancy while losing 11.9 points
of beneficial self-correction. For the two certified models, where no
rewrite fires, progressive sycophancy rises on both datasets. Where the
rewrite is applied, models become less likely to revise in either
direction, not more accurate. The calibrated component never engages selectively for any model, so the
```
New:
```latex
Table~\ref{tab:cert}. For the four uncertified models, progressive
sycophancy falls on both datasets (by 0.4 to 11.9 points), in several cases
by more than regressive sycophancy does: Gemma-3 1B on MedQuAD avoids 2.1
points of regressive sycophancy while losing 11.9 points of beneficial
self-correction. A withheld answer cannot count as progressive either, so
for these models CCR's final answers move less in both directions rather
than becoming more accurate. For the two certified models progressive
sycophancy rises on both datasets (by 0.5 to 12.0 points). Because the two
groups differ in more than certification (instruction tuning, initial
accuracy, and whether the rewrite runs), we do not attribute the contrast
to any one of these. The calibrated component never engages selectively for any model, so the
```
- **Risk / tradeoff:** +45 words. If length is tight, drop the sentence beginning "Because the two groups differ".

---

### R8: Oracle paragraph wording depends on which rerun is reported

- **Priority:** Major. **Category:** correctness, reviewer-critique. **Location:** §5.4 (l.989–991, l.1032–1037, l.1039–1040).
- **Problem:**
  - Table 4's HealthSearchQA oracle value for Gemma-3 1B is 0.168, but the other of the two reruns gave 0.124. "Stays well above the 0.10 target" and "improve only by 0.007–0.031" are true for the reported run only (0.124 is an improvement of 0.063). The conclusion survives: no uncertified HealthSearchQA model reaches 0.10 in either run.
  - The noise floor "about 0.05" sits just above the largest observed rerun difference (0.044) and just above the larger MedQuAD crossing (0.045). It reads as chosen to cover the crossing. Saying the crossings are comparable to rerun variation makes the same point without a threshold.
  - The side experiment was MedQuAD-only, with 30 calibration questions out of 60 items. "60-item side experiment (240 calibration instances)" suggests 60 calibration items.
- **Evidence:** `logs/error_oracle1khs_gemma_1b_*.txt` contain `alpha_min=0.124` and `alpha_min=0.168`; rerun differences per pair from the `logs/error_oracle1k*` files (seven identical; three ≤ 0.007; HealthSearchQA Phi-1.5 0.048/0.066; HealthSearchQA Gemma-3 1B 0.168/0.124); `docs/FINAL_DIAGNOSTICS_TRACKER.md:122–135, 280–293`; `slurm/verify_gptoss_scorer_*.slurm` (`--max_items 60 --calib_frac 0.5 --domain medquad`).
- **Proposed change:**

(a)

Old:
```latex
(Table~\ref{tab:oracle}). All six $\alpha_{\min}$ values improve, but only
by $0.007$--$0.031$: the nearest uncertified model (Gemma-3 1B, $0.168$)
stays well above the $0.10$ target, and the two furthest barely move
```
New:
```latex
(Table~\ref{tab:oracle}). All six $\alpha_{\min}$ values improve, but by at
most $0.03$: the nearest uncertified model (Gemma-3 1B, $0.168$; $0.124$ in
a second run, see below) stays above the $0.10$ target, and the two furthest barely move
```

(b)

Old:
```latex
reruns, three differed by at most $0.007$, and two by up to $0.044$, so we
treat changes below about $0.05$ as noise. Every change in
Table~\ref{tab:oracle}, including both MedQuAD crossings ($0.045$ and
$0.030$), falls below this floor. The HealthSearchQA result does not depend
on it: it rests on the failure to certify, not on the size of any shift.
```
New:
```latex
reruns, three differed by at most $0.007$, and two by $0.018$ and $0.044$
(Phi-1.5 and Gemma-3 1B on HealthSearchQA; Table~\ref{tab:oracle} reports
one of the two runs). Both MedQuAD crossings ($0.045$ and $0.030$) are
therefore comparable to the variation between identical runs, and we treat
them as noise. The HealthSearchQA result does not depend on this: no
uncertified model reaches $0.10$ in either run.
```

(c)

Old:
```latex
A separate 60-item side experiment (240 calibration instances) points the
same way. Revising the scoring prompt left ranking near or below chance
```
New:
```latex
A separate MedQuAD side experiment (30 calibration questions, 240
instances) points the same way. Revising the scoring prompt left ranking near or below chance
```
- **Risk / tradeoff:** +25 words. Table 4's caption says "within run-to-run noise (see text)", which stays correct. The Conclusion's "no HealthSearchQA model becomes certifiable" stays correct.

---

### R9: The Conclusion generalises beyond what was tested

- **Priority:** Major. **Category:** reviewer-critique (overclaiming). **Location:** Conclusion (l.1117–1120); Abstract (l.176–179).
- **Problem:** "Conformal risk control does not provide a deployable safeguard against sycophancy in medical question answering" is a claim about the method in general. The evidence is one 7B scorer, six models of at most 4B parameters, and one fallback threshold. The paper's own §5.4 says the scorer is the bottleneck, which implies a better scorer might work. "These results establish…" in the abstract has the same problem in milder form.
- **Evidence:** §4.2 (single Qwen2.5-7B scorer), Limitations (l.1112–1114).
- **Proposed change:**

(a) Conclusion

Old:
```latex
may access a reference answer, conformal risk control does not provide a
deployable safeguard against sycophancy in medical question answering:
only two of six models certify at a 10\% target, the two base Phi models,
```
New:
```latex
may access a reference answer, conformal risk control with a Qwen2.5-7B
scorer did not provide a deployable safeguard against sycophancy for the
six small models we tested:
only two of six models certify at a 10\% target, the two base Phi models,
```

(b) Abstract

Old:
```latex
results establish conformal calibration as an audit instrument for detecting
when LLM-based risk scoring cannot support selective intervention, and
```
New:
```latex
results show how conformal calibration can serve as an audit instrument
for detecting when LLM-based risk scoring cannot support selective intervention, and
```
- **Risk / tradeoff:** the conclusion is less quotable but defensible. Update the CMT abstract after (b).

---

### R10: Related Work says no mitigation acts on outputs; the novelty claim needs two citations

- **Priority:** Major. **Category:** reviewer-critique (related work). **Location:** §2.1 (l.305–306); §2.2 (l.352–354).
- **Problem:**
  - "Existing approaches act on the model rather than on its outputs" is contradicted by a paper already cited: Chen et al. (2025) test prompt-only mitigations alongside fine-tuning.
  - "First to evaluate conformal risk control against sycophancy" is safer when it names the nearest work. Two papers are close: one calibrates uncertainty estimates under user suggestions, and one shows that peer pressure from other agents breaks conformal coverage. Neither applies conformal risk control or claim filtering to user-induced sycophancy, so the claim stands once narrowed.
- **Evidence:** Chen et al., npj Digital Medicine 2025, doi 10.1038/s41746-025-02008-z (prompt-based rejection and recall hints; https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12534679/). Sicilia, Inan, Alikhani, "Accounting for Sycophancy in Language Model Uncertainty Estimation", arXiv:2410.14746 (I confirmed the page; the Findings of NAACL 2025 venue is reported by a search and not confirmed by me). Hu and Su, "Conformity Breaks Conformal Prediction", arXiv:2609.04445, 3 Sep 2026 (I confirmed the page: coverage falls from 90% to 74% under unanimous wrong peers).
- **Proposed change:**

(a) §2.1

Old:
```latex
Mitigation has received less attention, and existing approaches act on the
model rather than on its outputs. Wei et al.~\cite{wei2023simple}
```
New:
```latex
Mitigation has received less attention. Prompt-level instructions help in
some settings~\cite{chen2025helpfulness} but carry no statistical
guarantee, and most other approaches act on the model rather than on its
outputs. Wei et al.~\cite{wei2023simple}
```

(b) §2.2

Old:
```latex
reasoning without external feedback~\cite{huang2024large}. To our
knowledge, this work is the first to evaluate conformal risk control
against sycophancy, using calibration both as a safeguard and as an audit.
```
New:
```latex
reasoning without external feedback~\cite{huang2024large}. Closest to our
setting, Sicilia et al.~\cite{sicilia2025accounting} adjust uncertainty
estimates for user suggestions, and Hu and Su~\cite{hu2026conformity} show
that peer pressure from other agents breaks conformal coverage in
multiple-choice question answering. To our knowledge, this work is the
first to evaluate conformal risk control and claim filtering as a defense
against user-induced sycophancy, using calibration both as a safeguard and
as an audit.
```

Add to the Overleaf `.bib` (export the first from the ACL Anthology if the NAACL record exists; otherwise use it as an arXiv entry):
```bibtex
@article{sicilia2025accounting,
  title={Accounting for Sycophancy in Language Model Uncertainty Estimation},
  author={Sicilia, Anthony and Inan, Mert and Alikhani, Malihe},
  journal={arXiv preprint arXiv:2410.14746},
  year={2024}
}
@article{hu2026conformity,
  title={Conformity Breaks Conformal Prediction},
  author={Hu, Yibo and Su, Hanyu},
  journal={arXiv preprint arXiv:2609.04445},
  year={2026}
}
```
- **Risk / tradeoff:** +55 words and two references (references do not count toward the page limit). If you apply only (a), nothing else is needed. If you would rather not cite a two-month-old preprint, keep the Sicilia sentence and drop the Hu and Su clause.

---

### R11: Say that judge labels were not validated against human annotation

- **Priority:** Major. **Category:** reviewer-critique (missing limitation). **Location:** Limitations (l.1076–1077).
- **Problem:** Every result depends on Qwen2.5-7B's correct / incorrect / erroneous labels. The Limitations discuss one model in several roles but never say whether the judge was checked against people. A TRUST reviewer on the evaluation track will ask.
- **Evidence:** I found no record of a human-labelled check anywhere under `docs/` (searched for human annotation, manual labelling, judge agreement, judge accuracy). **Confirm this is true before applying**; if any manual spot-check was done, describe it instead.
- **Proposed change:**

Old:
```latex
deployed performance. Qwen2.5-7B serves as judge, rebuttal
```
New:
```latex
deployed performance. Judge labels were not validated against human
annotation. Qwen2.5-7B serves as judge, rebuttal
```
- **Risk / tradeoff:** it invites the question, but an undisclosed gap is worse than a disclosed one.

---

### R12: Add question-level bootstrap intervals for Table 3's regressive differences

- **Priority:** Major. **Category:** reviewer-critique (statistics). **Location:** §5.3 first paragraph (l.776–781); Limitations (l.1085–1089). **Needs HPC.**
- **Problem:** "CCR lowers regressive sycophancy for four of six / five of six models" counts differences as small as 1.5 points with no uncertainty. With eight correlated drafts per question the effective sample is about 800 questions. The paper admits it reports no intervals. This is cheap to fix: the data exist and no model has to run.
- **Is it worth doing?** Yes, for the regressive differences only. The main conclusions (certification fails; 26.3% vs 4.9%) do not need intervals. The count of "models improved" does.
- **Evidence / how to run:** `docs/revision_ci.py` (stdlib only, read-only, about 1–2 minutes on the login node; tested on a synthetic fixture, not yet on real data). It uses the same record fields that `docs/paper_v2_recompute.py` already ran on.

```bash
cd ~/SycEval_implementation
git pull
python3 docs/revision_ci.py | tee revision_ci_output.txt
```
Paste back `revision_ci_output.txt`. Section A gives each Table 3 difference with a 95% interval (a `*` marks intervals that exclude zero); section B gives the outcome split behind 26.3% vs 4.9% for every model; section C is for R21.
- **Proposed change:** apply (a) **instead of R17** (same paragraph).

(a) §5.3

Old:
```latex
Despite failed calibration for four models, CCR lowers regressive
sycophancy end-to-end for most of them (Table~\ref{tab:e2e}): four of six
models on MedQuAD and five of six on HealthSearchQA. No model's effect
changes sign between datasets, and effects are larger on HealthSearchQA,
although several differences are within a few points and are not tested for
significance. LLaMA-3.2 1B is the only model made worse, on both datasets.
```
New:
```latex
Although calibration fails for four of the six models, CCR lowers
regressive sycophancy end-to-end for most models (Table~\ref{tab:e2e}):
four of six on MedQuAD and five of six on HealthSearchQA. Under a
question-level bootstrap (resampling questions, 2{,}000 replicates), the
95\% interval excludes zero for {{FILL: the model--dataset pairs whose
d_regr carries '*' in section A, with intervals, e.g. "Gemma-3 4B on both
datasets ($-10.2$ [$lo$, $hi$] and $-21.1$ [$lo$, $hi$])"}}; the
remaining differences are within sampling noise. No model's effect changes
sign between datasets. LLaMA-3.2 1B is the only model made worse, on both
datasets{{FILL: add ", though its interval includes zero" if section A shows no '*'}}.
```

(b) Limitations

Old:
```latex
than on MedQuAD, whose references are institutionally curated. We do not report
confidence intervals or pairwise significance tests; with eight correlated
drafts per question, the effective sample size is closer to the number of
questions than the number of instances, so differences of a few points should be
read with caution.
```
New:
```latex
than on MedQuAD, whose references are institutionally curated. We report
question-level bootstrap intervals only for the regressive differences in
Table~\ref{tab:e2e} (\S\ref{sec:e2e}); with eight correlated drafts per
question, the effective sample size is closer to the number of questions
than the number of instances, so other differences of a few points should be
read with caution.
```
- **Risk / tradeoff:** some "improvements" will probably fall inside noise, and the headline count softens. That is more defensible, and it strengthens the paper's point that the apparent gains are not solid. If the cluster is unavailable before the deadline, apply R17 and leave the Limitations sentence as it is.

---

### R13: Add CCS concepts

- **Priority:** Major. **Category:** compliance. **Location:** front matter (l.182–187).
- **Problem:** The ACM template expects CCS concepts and the file still has a TODO. The author guidelines require ACM formatting conventions.
- **Evidence:** `sigconf.tex` l.182–187; https://eigtrust.acm.org/trust2027/authors-guidelines/. The concept ids below are from memory (dl.acm.org blocked automated access), so regenerate the block at https://dl.acm.org/ccs to be safe; the three concept names are standard.
- **Proposed change:**

Old:
```latex
%% TODO: Generate real CCS concepts at https://dl.acm.org/ccs.cfm and paste them here, e.g.:
%% \begin{CCSXML}
%% ...
%% \end{CCSXML}
%% \ccsdesc[500]{...}
%%
```
New:
```latex
\begin{CCSXML}
<ccs2012>
   <concept>
       <concept_id>10010147.10010178.10010179</concept_id>
       <concept_desc>Computing methodologies~Natural language processing</concept_desc>
       <concept_significance>500</concept_significance>
   </concept>
   <concept>
       <concept_id>10010405.10010444.10010449</concept_id>
       <concept_desc>Applied computing~Health informatics</concept_desc>
       <concept_significance>300</concept_significance>
   </concept>
   <concept>
       <concept_id>10010147.10010341.10010342.10010345</concept_id>
       <concept_desc>Computing methodologies~Uncertainty quantification</concept_desc>
       <concept_significance>300</concept_significance>
   </concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computing methodologies~Natural language processing}
\ccsdesc[300]{Applied computing~Health informatics}
\ccsdesc[300]{Computing methodologies~Uncertainty quantification}
```
- **Risk / tradeoff:** adds about three lines under the abstract on page 1.

---

### R14: Split contribution 3 into two items

- **Priority:** Minor. **Category:** clarity. **Location:** §1 (l.270–276). **Includes R1(b).**
- **Problem:** Contribution 3 holds two unrelated findings: `α_min` as a diagnostic, and the withholding result. The withholding result is the paper's most distinctive finding and is easy to miss inside a long item.
- **Proposed change:** (if R1(b) is already applied, the third line of Old reads "smallest"; the rest is unchanged)

Old:
```latex
    report per-model $\alpha_{\min}$---the loosest error target at which any
    certifiable region exists---replacing coarse threshold sweeps with a single summary quantity that also measures how far an uncertifiable model
    sits from certifiability. We further show that the reduction observed end-to-end is consistent with
    claim filtering, the one component carrying no statistical guarantee,
    which lowers regressive sycophancy largely by withholding answers.
```
New:
```latex
    report per-model $\alpha_{\min}$---the smallest error target at which any
    certifiable region exists---replacing coarse threshold sweeps with a single summary quantity that also measures how far an uncertifiable model
    sits from certifiability.

    \item We show that the reduction observed end-to-end is consistent with
    claim filtering, the one component carrying no statistical guarantee,
    which lowers regressive sycophancy largely by withholding answers
    rather than correcting them.
```
- **Risk / tradeoff:** none.

---

### R15: The abstract's 4.9% does not appear in the body

- **Priority:** Minor. **Category:** consistency. **Location:** §5.3 (l.790–791); Abstract (l.175).
- **Problem:** The abstract compares 26.3% with "4.9% without it", but the body gives only the CCR split and the baseline's 77.5%. A reader cannot find the 4.9. The abstract also says "no answer" where the body says "no judgeable answer"; the 26.3% is the judge's erroneous label, of which 24.2 points are literally the empty placeholder.
- **Evidence:** The 26.3 / 53.3 / 20.4 split is in the project memory note (HPC recompute of 2026-10-18) and commit `bee705d`; 53.3 is also in `logs/` (job 144677). The 4.9 has no saved output in the repo; it came from an earlier cluster run. `docs/revision_ci.py` section B re-derives all of these (expect Gemma-3 4B HealthSearchQA: base err = 4.9, ccr err = 26.3, withheld ≈ 24.2). From the `find_case_examples.py` counts you ran: 1,266 of 5,240 initially correct answers end as the placeholder (24.2%).
- **Proposed change:**

(a) §5.3

Old:
```latex
HealthSearchQA answers, 77.5\% end incorrect at baseline; under CCR,
```
New:
```latex
HealthSearchQA answers, 77.5\% end incorrect and 4.9\% end with no
judgeable answer at baseline; under CCR,
```

(b) Abstract

Old:
```latex
26.3\% of initially correct answers end with no answer under CCR, versus
```
New:
```latex
26.3\% of initially correct answers end with no judgeable answer under CCR, versus
```
- **Risk / tradeoff:** if `revision_ci.py` shows a baseline value other than 4.9, fix both places. Update the CMT abstract after (b).

---

### R16: Define "erroneous" as the judge prompt does

- **Priority:** Minor. **Category:** method-code. **Location:** §3.1 (l.407–408).
- **Problem:** The paper says erroneous means "a refusal or an output that cannot be judged". The judge prompt says refusal or "a response on a wildly different topic"; the label is also assigned when the judge's own output cannot be parsed.
- **Evidence:** `judge.py:23`, `judge.py:61–74`.
- **Proposed change:**

Old:
```latex
that becomes \emph{erroneous} (a refusal or an output that cannot be
judged) is not counted as regressive: it may be a separate reliability
```
New:
```latex
that becomes \emph{erroneous} (a refusal, an off-topic response, or an
output the judge cannot classify) is not counted as regressive: it may be a separate reliability
```
- **Risk / tradeoff:** none.

---

### R17: Ambiguous "most of them" in §5.3 (use only if R12 is not applied)

- **Priority:** Minor. **Category:** clarity. **Location:** §5.3 (l.776–777).
- **Problem:** "Despite failed calibration for four models, CCR lowers regressive sycophancy end-to-end for most of them: four of six…" "Them" reads as the four uncertified models, but the count is over all six. Among the four uncertified models the count is two of four on MedQuAD and three of four on HealthSearchQA.
- **Proposed change:**

Old:
```latex
Despite failed calibration for four models, CCR lowers regressive
sycophancy end-to-end for most of them (Table~\ref{tab:e2e}): four of six
```
New:
```latex
Although calibration fails for four of the six models, CCR lowers
regressive sycophancy end-to-end for most models (Table~\ref{tab:e2e}): four of six
```
- **Risk / tradeoff:** none. R12(a) rewrites the whole paragraph and already contains this fix.

---

### R18: Reproducibility details and a code-availability line

- **Priority:** Minor. **Category:** compliance (reproducibility), method-code. **Location:** §4.2 (l.626–627).
- **Problem:** The CFP asks that evaluation choices be reported clearly enough to reproduce. The paper gives no seed, no generation cap (the figure's initial answer is cut off at the cap), and no statement about code. "Used directly without preprocessing" also contradicts §4.1, which describes a processed release and removed duplicates.
- **Evidence:** seed 7 (`data_split.seed` in all 12 thresholds files); `max_new_tokens=256` (`models.py:299, 367`), 512 for the risk scorer (`conformal_v2/syco_risk.py:152`).
- **Proposed change:**

Old:
```latex
slightly (Section~\ref{sec:oracle}). Raw question-answer pairs are used
directly without preprocessing or prompt augmentation.
```
New:
```latex
slightly (Section~\ref{sec:oracle}). Questions are passed to the models
verbatim, without prompt augmentation; generations are capped at 256 new
tokens (512 for the risk scorer), and the sampling seed is 7. Code,
prompts, and per-instance outputs will be released upon publication.
```
- **Risk / tradeoff:** +30 words. The last sentence is a commitment; delete it if you and your co-authors have not agreed to release.

---

### R19: The identical-prompt check is stated more broadly than it was run

- **Priority:** Minor. **Category:** correctness. **Location:** Limitations (l.1098–1100).
- **Problem:** "Changes no sign in Table 3" covers all three columns, but the check was on the regressive differences. It also hides that one effect nearly disappears.
- **Evidence:** project memory note of the HPC recompute, 2026-10-18: all 12 regressive differences keep their sign on the five identical-prompt slots; magnitudes move by at most about 1.5 points except LLaMA-3.2 3B on HealthSearchQA, −3.5 → −0.7. The output is not saved in the repo (see §4).
- **Proposed change:**

Old:
```latex
the five draft slots with identical prompts changes no sign in
Table~\ref{tab:e2e}. A few MedQuAD questions share identical text; in
```
New:
```latex
the five draft slots with identical prompts changes the sign of no
regressive difference in Table~\ref{tab:e2e}, although LLaMA-3.2 3B's
HealthSearchQA reduction shrinks from $3.5$ to $0.7$ points. A few MedQuAD questions share identical text; in
```
- **Risk / tradeoff:** +15 words.

---

### R20: The certified models also lose 20–29% of initially correct answers to withholding

- **Priority:** Minor. **Category:** reviewer-critique (strengthens the argument). **Location:** §5.2 (l.722–725).
- **Problem:** §5.2 explains the Phi certifications by low initial accuracy alone. The cluster counts show the claim filter also empties a fifth to nearly a third of these models' initially correct answers, and an emptied answer cannot count against the bound. That ties the certification result to the withholding result.
- **Evidence:** the `find_case_examples.py` counts table from your cluster run (answers whose final text is the placeholder, among initially correct): Phi-1.5 MedQuAD 111/552 = 20.1%; Phi-2 MedQuAD 196/696 = 28.2%; Phi-1.5 HealthSearchQA 329/1,208 = 27.2%; Phi-2 HealthSearchQA 341/1,176 = 29.0%. That output is in the session transcript, not in the repo; `docs/revision_ci.py` section B prints the same shares ("withheld", phi rows).
- **Proposed change:**

Old:
```latex
only if the initial answer was correct, few instances count against the
bound, so their certification reflects low initial accuracy, not
robustness to pressure.
```
New:
```latex
only if the initial answer was correct, few instances count against the
bound, and claim filtering empties 20--29\% of their initially correct
test answers, which then cannot count as regressive either. Their
certification therefore reflects low initial accuracy and withheld answers,
not robustness to pressure.
```
- **Risk / tradeoff:** +18 words. The shares are from the test split; the bound is fitted on calibration data, where the same filter runs, so the statement is about the mechanism, not a calibration count.

---

### R21 (optional): How much the best threshold gains over accepting every draft

- **Priority:** Minor, optional. **Category:** reviewer-critique. **Location:** §5.2 (l.769–770). **Needs HPC.**
- **Problem:** §5.2 notes that `α_min` tracks baseline regressive rates but does not say why that matters: if the risk score ranked drafts well, `α_min` would fall far below the accept-everything rate. Reporting that gap shows the scorer's weakness from the deployed data, before the oracle section.
- **Evidence / how to run:** `docs/revision_ci.py` section C (same run as R12). For each model it prints `alpha_min`, where it is attained (`n*` of the calibration instances), the accept-everything value, and the gap.
- **Proposed change:**

Old:
```latex
models that pass on both. Section~\ref{sec:oracle} localizes the cause of
these failures.
```
New:
```latex
models that pass on both. For the four uncertified models, the best
threshold improves on accepting every draft by only {{FILL: min--max of
"gain over accept-all" over the eight uncertified rows of section C}}, so
the risk score contributes little separation. Section~\ref{sec:oracle}
localizes the cause of these failures.
```
- **Risk / tradeoff:** +25 words. Apply only if the gaps are small (a few hundredths). If a gap is large but attained on a small accepted set, report that instead: "the minimum is attained on only n* of 1,600 drafts".

---

### R22: Four one-line precision fixes (independent)

- **Priority:** Minor. **Category:** correctness, clarity.

(a) §2.1, FlipFlop (l.295). The challengers in that study include "Are you sure?" but are not limited to it (arXiv:2311.08596).

Old:
```latex
experiment~\cite{laban2023flipflop} shows that a simple ``Are you sure?''
```
New:
```latex
experiment~\cite{laban2023flipflop} shows that a simple challenge such as ``Are you sure?''
```

(b) §3.2.1, Wilson bound (l.453–454). The code uses z = 1.96, the upper limit of a two-sided 95% interval (`conformal_v2/conformal_thresholds.py:63`).

Old:
```latex
$\tau_{\mathrm{claim}}$: the most permissive threshold for which the 95\%
Wilson upper bound on the rate of unsupported claims among retained
```
New:
```latex
$\tau_{\mathrm{claim}}$: the most permissive threshold for which the upper
limit of the two-sided 95\% Wilson interval on the rate of unsupported claims among retained
```

(c) §3.1, in-context history (l.420–421). The history is one plain-text prompt that grows, and it carries the unfiltered drafts (`run_conformal_v2.py:200–224`).

Old:
```latex
rebuttals are presented in sequence within one conversation, so later
drafts are conditioned on earlier rebuttals and drafts; preemptive drafts
```
New:
```latex
rebuttals are presented in sequence as one growing plain-text transcript, so later
drafts are conditioned on earlier rebuttals and unfiltered drafts; preemptive drafts
```

(d) §5.1, progressive comparison (l.647–648). Phi-1.5 is 18.5 vs 18.4, a 0.1-point difference counted as "lower".

Old:
```latex
MedQuAD for all six models, while progressive sycophancy is lower on
HealthSearchQA for five of six (all but Phi-2). These rates are
```
New:
```latex
MedQuAD for all six models, while progressive sycophancy is lower on
HealthSearchQA for four of six, unchanged for Phi-1.5 (18.5 vs.\ 18.4), and
higher for Phi-2. These rates are
```
- **Risk / tradeoff:** none; about +15 words in total.

---

### R23: Delete the commented-out duplicate of Table 3

- **Priority:** Minor. **Category:** clarity (source hygiene). **Location:** l.887–934.
- **Problem:** A commented-out earlier version of Table 3 sits between the figure and the live table. It does not affect the PDF, but it repeats the caption text and `\label{tab:e2e}`, which makes find-and-replace edits error-prone.
- **Proposed change:** delete the whole commented block. It starts at the first line below and ends at the second (both inclusive; every line between them begins with `%` or is blank). The live `\begin{table}[t]` that follows stays.

First line (l.887), which is the only line in the file reading exactly this:
```latex
% \begin{table}[t]
```
Last line (l.934), which is the only line in the file reading exactly this:
```latex
% \end{table}
```
- **Risk / tradeoff:** none. This is the one entry given as a line range instead of an Old/New pair, because the block is 48 lines.

---

### R24–R27: Length offsets (apply only if the paper is over 9 pages)

All four remove repetition and change no claim.

**R24** — §3.2 preamble (l.429–433), repeated by the Method introduction. −35 words.

Old:
```latex
CCR uses two complementary signals. Claim-level scoring identifies
which factual statements should be retained, while answer-level risk
scoring estimates whether the response exhibits evidence of an
unjustified shift following the rebuttal. The two components are
calibrated separately and serve different roles in the intervention.
```
New:
```latex
CCR's claim-level and answer-level components are calibrated separately.
```

**R25** — §3.2.2 raw-draft paragraph (l.488–494). −40 words.

Old:
```latex
filtering}. We use the same raw-draft representation during both
calibration and testing so that the calibrated score function matches
the one applied at inference time. Scoring before filtering also
preserves signals such as hedging, qualification, or explicit
acknowledgment of the rebuttal that could disappear after
low-validity claims are removed.
```
New:
```latex
filtering}, in both calibration and testing, which preserves signals such
as hedging or explicit acknowledgment of the rebuttal that filtering could
remove.
```

**R26** — Discussion, oracle paragraph (l.1066–1069), restated in the Conclusion (l.1123–1125). −45 words. Also remove one of the blank lines around it.

Old:
```latex
The oracle diagnostic (\S\ref{sec:oracle}) places the bottleneck in the risk
scorer, or in label noise, rather than in the models: even with the reference
answer, Qwen2.5-7B cannot separate regressive from non-regressive drafts well
enough to certify any additional model on HealthSearchQA.
```
New:
```latex
```

**R27** — §5.2 ordering paragraph (l.761–766). −45 words.

Old:
```latex
Certifiability closely tracks baseline regressive rates. On MedQuAD,
$\alpha_{\min}$ orders the six models exactly as their baseline regressive
rates in Table~\ref{tab:baseline} do; on HealthSearchQA the ordering is
preserved up to a tie between the two 1B models (both $0.187$) and a swap
of the two Phi models, whose baseline rates differ by under one point. $\alpha_{\min}$ is higher on HealthSearchQA for five of six
models, consistent with the elevated baseline rates there. Phi-1.5
```
New:
```latex
Certifiability closely tracks baseline regressive rates: $\alpha_{\min}$
orders the six models as their baseline rates in Table~\ref{tab:baseline}
do, exactly on MedQuAD and up to near-ties on HealthSearchQA, and it is
higher on HealthSearchQA for five of six models. Phi-1.5
```

## 4. Unverifiable claims

These could not be traced to a file in the repo. None looks wrong; each lacks a saved source.

| Claim in the paper | Where the number came from | What would verify it |
|---|---|---|
| Table 3 ΔErr column (12 cells); "13.4 and 18.1", "7.2 and 7.5" (l.786–787) | Memory note of the 2026-10-18 cluster recompute; commit `bee705d` message (four of the twelve) | `python3 docs/revision_ci.py` section A (`d_err`) |
| 26.3% / 20.4% (CCR) and 4.9% (baseline) for Gemma-3 4B HealthSearchQA (abstract, l.791–793) | Same memory note; 4.9 only from an earlier cluster run. 53.3 and 77.5 are verified | `docs/revision_ci.py` section B |
| "9–31% of drafts" emptied by the filter (l.789–790) | Memory note only | `python3 docs/paper_v2_recompute.py \| tee docs/paper_v2_recompute_output.txt`, then commit the output |
| Scorer "emits only 12–16 distinct values" (l.1031) | Memory note only | Same recompute script, section B |
| Identical-prompt slots change no sign (l.1098–1100) | Memory note only | Same recompute script, section F |
| Figure: initial answers, rebuttals, claim scores | Output of rescore job 162625, pasted in the session; `results/case_claims_{a,b}.json` exist only on the cluster. Risk scores 0.95 and 1.00 are confirmed from your `find_case_examples.py` output | Copy the two JSON files into `docs/` and commit them |
| `\cite{yang2024}` is the Qwen2.5 report | The full `.bib` is only in Overleaf | Check the entry is "Qwen2.5 Technical Report", arXiv:2412.15115 (not the Qwen2 report, arXiv:2407.10671) |
| `\cite{jeong2024olaph}` venue | `references_additions.bib` says Findings of EMNLP 2024, which I could not confirm; a co-author page says NAACL 2025 | Export the entry from the ACL Anthology, or cite arXiv:2405.12701 |
| "The full MedQuAD collection contains roughly 47,000" (l.589–590) | External fact; the MedQuAD paper reports 47,457 | Nothing needed |
| Phi-1.5 = 1.3B, Phi-2 = 2.7B parameters | Model cards; not recorded in the repo | Nothing needed |
| No human validation of the judge (asserted by R11) | Absence of any record under `docs/` | Your confirmation |

One practical suggestion: `docs/paper_v2_recompute.py` has run successfully on the cluster before, but its output was never saved. Running it once with `tee` and committing the text file closes most of this table.

## 5. Not proposed

- **Reinstating the rewrite analysis or Table 4, or un-merging Discussion / Limitations / Conclusion.** Decided already.
- **Changing the figure.** Its content matches the cluster rerun. Its level names are handled by R5(a) in the text.
- **New experiments or baselines** (a second scorer, a prompt-only mitigation arm, human labels). No time before 15 November, and no GPU reruns were asked for; R9 and R11 disclose the gaps instead.
- **Renaming "certify" throughout.** R3 defines it once, which is enough.
- **Removing the "Rewrite" column from Table 2 or the † notes in Table 3.** They describe what ran.
- **Intervals for Table 1 or `α_min` in the text.** `revision_ci.py` prints them, but adding them costs space and no claim depends on them.
- **Noting that the HealthSearchQA "tie" at 0.187 is an inversion at four decimals** (0.18699 vs 0.18721). Too fine to matter.
- **Adding "for PaLM models" to the Wei et al. sentence.** Accurate but not needed.
- **Stating that SycEval excludes erroneous responses while this paper keeps them in every denominator.** Probably true (reported by a literature check quoting "15345 non-erroneous responses"), but I did not read the SycEval methods section myself, and the paper's definitions are already explicit.
- **Disclosing that a claim-score parse failure defaults to 0.5 (kept) and that the rewrite still runs on an emptied draft.** Both are true in the code (`scorers.py:9–14`, `conformal_v2/safe_rewrite.py:106–129`) but are implementation details that would add length without changing a result.
- **Removing the scorer prompts' leftover "Ground Truth (if provided)" wording.** That is a code matter; no reference is ever supplied, so the paper's reference-blind claim holds.
- **Anonymity changes.** None needed: no names, affiliations, cluster name, or repository links appear in the text or comments; `anonymous` is set; acknowledgments are commented out.
