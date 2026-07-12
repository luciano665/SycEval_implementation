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

    def fake_rewrite(tested_model, question, rebuttal, draft_answer,
                     initial_answer, truth=None, **k):
        c.rewrite_truths.append(truth)
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


def make_cfg(oracle: bool) -> EvalConfig:
    return EvalConfig(backend="hf", rebuttal_strengths=("simple",),
                      oracle_truth=oracle)


FIT = ThresholdFitResult(alpha=0.1, tau_global=0.5)


def test_test_apply_truth_free_when_oracle_off(cap):
    rc.test_apply(make_cfg(False), [ITEM], "scorer", FIT,
                  enable_rewrite=True, claim_threshold=0.5)
    assert cap.risk_truths and all(t is None for t in cap.risk_truths)
    assert cap.claim_truths and all(t is None for t in cap.claim_truths)
    assert cap.rewrite_truths and all(t is None for t in cap.rewrite_truths)
    # Judges are evaluation-side: they ALWAYS get the reference answer.
    assert cap.judge_truths and all(t == "REFERENCE_ANSWER" for t in cap.judge_truths)


def test_test_apply_truth_passed_when_oracle_on(cap):
    rc.test_apply(make_cfg(True), [ITEM], "scorer", FIT,
                  enable_rewrite=True, claim_threshold=0.5)
    assert cap.risk_truths and all(t == "REFERENCE_ANSWER" for t in cap.risk_truths)
    assert cap.claim_truths and all(t == "REFERENCE_ANSWER" for t in cap.claim_truths)
    assert cap.rewrite_truths and all(t == "REFERENCE_ANSWER" for t in cap.rewrite_truths)


def test_calibration_truth_free_when_oracle_off(cap):
    rc.calibration_collect(make_cfg(False), [ITEM], "scorer", claim_alpha=0.05)
    assert cap.claim_truths and all(t is None for t in cap.claim_truths)
    assert cap.risk_truths and all(t is None for t in cap.risk_truths)
    # Labelers keep truth even with oracle off:
    assert cap.support_truths and all(t == "REFERENCE_ANSWER" for t in cap.support_truths)
    assert cap.judge_truths and all(t == "REFERENCE_ANSWER" for t in cap.judge_truths)


def test_calibration_truth_passed_when_oracle_on(cap):
    rc.calibration_collect(make_cfg(True), [ITEM], "scorer", claim_alpha=0.05)
    assert cap.claim_truths and all(t == "REFERENCE_ANSWER" for t in cap.claim_truths)
    assert cap.risk_truths and all(t == "REFERENCE_ANSWER" for t in cap.risk_truths)


def test_cli_flag_default_and_negation():
    import argparse
    parser = argparse.ArgumentParser()
    rc.add_oracle_truth_args(parser)
    assert parser.parse_args([]).oracle_truth is True
    assert parser.parse_args(["--no_oracle_truth"]).oracle_truth is False
    assert parser.parse_args(["--oracle_truth"]).oracle_truth is True
