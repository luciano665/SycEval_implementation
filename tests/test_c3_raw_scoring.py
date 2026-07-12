import pytest

from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import ThresholdFitResult
from config import EvalConfig

ITEM = {"question": "What is X?", "answer": "REFERENCE_ANSWER"}


@pytest.fixture
def risk_inputs(monkeypatch):
    captured = {"draft_answers": [], "judged_texts": []}
    monkeypatch.setattr(rc, "ask_model", lambda *a, **k: "RAW_DRAFT_TEXT")
    monkeypatch.setattr(rc, "auto_proposed_answers", lambda *a, **k: "PROPOSED")
    # Purified reconstruction will be "claim one claim two" != RAW_DRAFT_TEXT
    monkeypatch.setattr(rc, "decompose_answer",
                        lambda answer, model, **k: ["claim one", "claim two"])
    monkeypatch.setattr(rc, "score_claim_sycophancy", lambda *a, **k: 0.9)

    def fake_risk(scorer_model, question, rebuttal, initial_answer,
                  draft_answer, truth=None, **k):
        captured["draft_answers"].append(draft_answer)
        return 0.5

    monkeypatch.setattr(rc, "sycophancy_risk_score", fake_risk)

    def fake_judge(judge_model, q, truth, ans, **k):
        captured["judged_texts"].append(ans)
        return "correct"

    monkeypatch.setattr(rc, "judge_local", fake_judge)
    monkeypatch.setattr(rc, "judge_claim_support", lambda *a, **k: True)
    return captured


def test_calibration_risk_scores_raw_draft(risk_inputs):
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)
    rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05)
    assert risk_inputs["draft_answers"], "risk scorer never called"
    assert all(d == "RAW_DRAFT_TEXT" for d in risk_inputs["draft_answers"])


def test_calibration_bad_label_still_judged_on_purified(risk_inputs):
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)
    rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05)
    # calibration judges drafts AFTER purification (delivered-on-accept object)
    assert "claim one claim two" in risk_inputs["judged_texts"]
    assert "RAW_DRAFT_TEXT" not in risk_inputs["judged_texts"][1:]  # [0] is the initial answer


def test_test_apply_still_scores_raw(risk_inputs):
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)
    fit = ThresholdFitResult(alpha=0.1, tau_global=0.9)
    rc.test_apply(cfg, [ITEM], "scorer", fit, enable_rewrite=False, claim_threshold=0.5)
    assert all(d == "RAW_DRAFT_TEXT" for d in risk_inputs["draft_answers"])
