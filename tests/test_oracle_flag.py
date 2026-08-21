"""Deployable-only invariant (conformal_v9): the oracle-truth option was
removed. The claim scorer, risk scorer, and rewrite must NEVER receive the
dataset reference answer, in either calibration or test. The judges
(judge_local / judge_claim_support) are evaluation-side labelers and MUST
still receive it."""

import pytest

from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import ThresholdFitResult
from config import EvalConfig

ITEM = {"question": "What is X?", "answer": "REFERENCE_ANSWER"}


class Capture:
    def __init__(self):
        self.claim_truths = []
        self.risk_truths = []
        self.rewrite_truths = []
        self.judge_truths = []
        self.support_truths = []


@pytest.fixture
def cap(monkeypatch):
    c = Capture()
    monkeypatch.setattr(rc, "ask_model", lambda *a, **k: "RAW_DRAFT_TEXT")
    monkeypatch.setattr(rc, "auto_proposed_answers", lambda *a, **k: "PROPOSED")
    monkeypatch.setattr(rc, "decompose_answer",
                        lambda answer, model, **k: ["claim one", "claim two"])

    def fake_claim_score(claim, question, judge_model, rebuttal=None,
                         initial_answer=None, truth=None, **k):
        c.claim_truths.append(truth)
        return 0.9

    monkeypatch.setattr(rc, "score_claim_sycophancy", fake_claim_score)

    def fake_risk(scorer_model, question, rebuttal, initial_answer,
                  draft_answer, truth=None, **k):
        c.risk_truths.append(truth)
        return 1.0  # always above tau below -> rewrite path exercised

    monkeypatch.setattr(rc, "sycophancy_risk_score", fake_risk)

    def fake_rewrite(tested_model, question, draft_answer, **k):
        # 'truth' must no longer be part of the rewrite interface at all.
        c.rewrite_truths.append(k.get("truth", None))
        return "REWRITTEN"

    monkeypatch.setattr(rc, "anti_sycophancy_rewrite", fake_rewrite)

    def fake_judge(judge_model, q, truth, ans, **k):
        c.judge_truths.append(truth)
        return "correct"

    monkeypatch.setattr(rc, "judge_local", fake_judge)

    def fake_support(claim, q, truth, judge_model, **k):
        c.support_truths.append(truth)
        return True

    monkeypatch.setattr(rc, "judge_claim_support", fake_support)
    return c


def make_cfg() -> EvalConfig:
    return EvalConfig(backend="hf", rebuttal_strengths=("simple",))


FIT = ThresholdFitResult(alpha=0.1, tau_global=0.5)


def test_test_apply_intervention_is_truth_free(cap):
    rc.test_apply(make_cfg(), [ITEM], "scorer", FIT,
                  enable_rewrite=True, claim_threshold=0.5)
    assert cap.risk_truths and all(t is None for t in cap.risk_truths)
    assert cap.claim_truths and all(t is None for t in cap.claim_truths)
    assert cap.rewrite_truths and all(t is None for t in cap.rewrite_truths)
    # Judges are evaluation-side: they ALWAYS get the reference answer.
    assert cap.judge_truths and all(t == "REFERENCE_ANSWER" for t in cap.judge_truths)


def test_calibration_intervention_is_truth_free(cap):
    rc.calibration_collect(make_cfg(), [ITEM], "scorer", claim_alpha=0.05)
    assert cap.claim_truths and all(t is None for t in cap.claim_truths)
    assert cap.risk_truths and all(t is None for t in cap.risk_truths)
    # Labelers keep truth:
    assert cap.support_truths and all(t == "REFERENCE_ANSWER" for t in cap.support_truths)
    assert cap.judge_truths and all(t == "REFERENCE_ANSWER" for t in cap.judge_truths)


def test_no_oracle_option_remains():
    # The config field and the CLI-flag helper must be gone entirely.
    assert not hasattr(EvalConfig(), "oracle_truth")
    assert not hasattr(rc, "add_oracle_truth_args")
