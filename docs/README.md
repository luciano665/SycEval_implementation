# docs/ index

Quick pointer to what each doc in this folder is for. Not a source of
information itself — just tells you where to look.

| Doc | What it's for | Updated |
|---|---|---|
| `RESEARCH_WORKFLOW.md` | How the pipeline works — protocol, architecture, module map, infra. Read this first if you're new. | Rarely (stable reference) |
| `EXPERIMENT_TRACKER.md` | Live status of the 36 real V6 suite jobs (superseded by v9 — kept as historical record). Job IDs, model/dataset/oracle mapping, results locations, analysis checklist. | Historical, frozen |
| `V9_EXPERIMENT_TRACKER.md` | Live status of the 12-job conformal_v9 overnight suite (deployable-only, leak-free rewrite, folding-only labels, α=0.10, N=300 MedQuad, Qwen doing every referee role) — the baseline suite. Job IDs, what changed from V6, a pre-registered prediction to check results against. | As jobs launch/complete |
| `V9_JUDGE_SPLIT_TRACKER.md` | Live status of the 12-job independent-judge suite — same protocol as v9n, but judge_model=Mistral-7B-Instruct-v0.3, independent of rebuttal/risk-scorer=Qwen. Tests whether judge self-preference bias changes the headline numbers. | As jobs launch/complete |
| `V10_EXPERIMENT_TRACKER.md` | Live status of the 6-job exact-CRC suite — same as v9n but `threshold_method=exact_crc` (finite-sample-valid bound, fixes a multiple-comparisons gap in the Wilson-based method), reusing v9n's baseline. | As jobs launch/complete |
| `FINAL_DIAGNOSTICS_TRACKER.md` | The two mentor-requested diagnostics before shifting to writing: oracle-assisted risk scorer (calibration only) and the rewrite-only effect analysis. Both complete — oracle rescued 2 of 4 failing models, rewrite-effect analysis overturned a previously-reported finding (Gemma-4B). | Complete |
| `FINAL_N1000_EXPERIMENT_TRACKER.md` | The paper's final, reported experiment — N=1000 (200 calib/800 test), all 6 models, exact_crc. Supersedes v9n's numbers as the primary result; also documents where a cheap pre-check projection turned out wrong once real data was collected. | Complete, headline analysis pending |
| `RISK_SCORER_PROMPT_LOG.md` | History of risk-scorer prompt iterations (OLD -> v2 -> v3 -> ...) — what changed each time, the test results, current findings. | As new prompt versions are tested |
| `REWRITE_POLICY_LOG.md` | Investigation into whether the rewrite step helps or hurts, per model — the Gemma-4B-benefits/Llama-3B-harmed finding, why it happens, proposed fixes, and the open circularity question on how to validate a selective policy without cheating. | As the investigation progresses |
| `MENTOR_DISCUSSION_2026-08-15.md` | Discussion prep for a research mentor meeting — the open judgment calls (not code bugs) in plain language, framed as questions. | One-off, dated |
| `TEAM_UPDATE_2026-08-05.md` | Snapshot: the code-review fix campaign (C1-C21) summary, sent to the team. | Frozen — don't edit, write a new dated one instead |
| `TEAM_UPDATE_2026-08-09.md` | Snapshot: real suite results + the calibration-failure bottleneck investigation, sent to the team. | Frozen — don't edit, write a new dated one instead |

**Convention going forward**: living/reference docs (tracker, prompt log,
research workflow) get updated in place. Team updates are point-in-time
snapshots — when it's time for another one, add a new `TEAM_UPDATE_<date>.md`
rather than editing an old one.
