# docs/ index

Quick pointer to what each doc in this folder is for. Not a source of
information itself — just tells you where to look.

| Doc | What it's for | Updated |
|---|---|---|
| `RESEARCH_WORKFLOW.md` | How the pipeline works — protocol, architecture, module map, infra. Read this first if you're new. | Rarely (stable reference) |
| `EXPERIMENT_TRACKER.md` | Live status of the 36 real V6 suite jobs — job IDs, model/dataset/oracle mapping, results locations, analysis checklist. | As jobs launch/complete |
| `RISK_SCORER_PROMPT_LOG.md` | History of risk-scorer prompt iterations (OLD -> v2 -> v3 -> ...) — what changed each time, the test results, current findings. | As new prompt versions are tested |
| `TEAM_UPDATE_2026-08-05.md` | Snapshot: the code-review fix campaign (C1-C21) summary, sent to the team. | Frozen — don't edit, write a new dated one instead |
| `TEAM_UPDATE_2026-08-09.md` | Snapshot: real suite results + the calibration-failure bottleneck investigation, sent to the team. | Frozen — don't edit, write a new dated one instead |

**Convention going forward**: living/reference docs (tracker, prompt log,
research workflow) get updated in place. Team updates are point-in-time
snapshots — when it's time for another one, add a new `TEAM_UPDATE_<date>.md`
rather than editing an old one.
