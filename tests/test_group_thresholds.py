import json

import pytest

from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import (
    ThresholdFitResult,
    choose_threshold,
    fit_threshold_by_group,
)

# Two groups: "easy" has clean data (100 accepted, 0 bad -> valid tau),
# "hard" is all-bad (no tau satisfies alpha -> -1.0 sentinel).
EASY_SCORES = [i / 1000 for i in range(100)]
EASY_BAD = [0] * 100
HARD_SCORES = [0.1, 0.2]
HARD_BAD = [1, 1]

SCORES = EASY_SCORES + HARD_SCORES
BAD = EASY_BAD + HARD_BAD
GROUPS = [("in-context", "simple", "correct")] * 100 + [("preemptive", "ethos", "correct")] * 2


def test_fit_thresholds_fits_groups_when_enabled():
    fit = rc.fit_thresholds(scores=SCORES, bad=BAD, groups=GROUPS,
                            alpha=0.05, use_group_thresholds=True)
    assert fit.tau_by_group is not None
    assert fit.tau_by_group[("in-context", "simple", "correct")] == max(EASY_SCORES)
    assert fit.tau_by_group[("preemptive", "ethos", "correct")] == -1.0  # kept, not dropped


def test_fit_thresholds_no_groups_when_disabled():
    fit = rc.fit_thresholds(scores=SCORES, bad=BAD, groups=GROUPS,
                            alpha=0.05, use_group_thresholds=False)
    assert fit.tau_by_group is None


def test_group_thresholds_roundtrip_and_selection():
    fit = rc.fit_thresholds(scores=SCORES, bad=BAD, groups=GROUPS,
                            alpha=0.05, use_group_thresholds=True)
    d = rc.threshold_to_json(fit, tau_claim=0.4, claim_alpha=0.05,
                             tau_claim_fallback=False, oracle_truth=True)
    # Must be JSON-serializable (tuple keys were the old bug)
    d2 = json.loads(json.dumps(d))
    fit2, *_ = rc.thresholds_from_json(d2)
    assert fit2.tau_by_group == fit.tau_by_group
    # choose_threshold must select the GROUP tau after a round-trip
    tau = choose_threshold(fit2, ("in-context", "simple", "correct"))
    assert tau == max(EASY_SCORES)
    # unknown group falls back to global
    assert choose_threshold(fit2, ("nope", "nope", "nope")) == fit2.tau_global


def test_legacy_stringified_group_dict_ignored_with_warning(caplog):
    legacy = {"alpha": 0.05, "tau_global": 0.7,
              "tau_by_group": {"('in-context', 'simple', 'correct')": 0.9},
              "tau_claim": 0.3, "claim_alpha": 0.05}
    with caplog.at_level("WARNING"):
        fit, *_ = rc.thresholds_from_json(legacy)
    assert fit.tau_by_group is None
    assert any("tau_by_group" in rec.message for rec in caplog.records)


def test_failed_group_logs_error(caplog):
    with caplog.at_level("ERROR"):
        rc.fit_thresholds(scores=SCORES, bad=BAD, groups=GROUPS,
                          alpha=0.05, use_group_thresholds=True)
    assert any("GROUP CALIBRATION FAILED" in rec.message for rec in caplog.records)


def test_fit_threshold_by_group_direct():
    by_group = fit_threshold_by_group(SCORES, BAD, GROUPS, alpha=0.05)
    assert set(by_group) == {("in-context", "simple", "correct"),
                             ("preemptive", "ethos", "correct")}
