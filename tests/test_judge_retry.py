import json

import pytest

import judge as judge_mod
from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import ThresholdFitResult
from config import EvalConfig

ITEM = {"question": "What is X?", "answer": "REFERENCE_ANSWER"}


# ---------------------------------------------------------------------------
# judge_local retry -> fallback behavior
# ---------------------------------------------------------------------------

def test_judge_local_succeeds_without_retry(monkeypatch):
    monkeypatch.setattr(judge_mod, "ask_model", lambda *a, **k: "correct")
    result = judge_mod.judge_local("model", "q", "truth", "answer")
    assert result == "correct"


def test_judge_local_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_ask_model(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return "garbage nonsense"
        return "incorrect"

    monkeypatch.setattr(judge_mod, "ask_model", fake_ask_model)
    result = judge_mod.judge_local("model", "q", "truth", "answer", retries=1)
    assert result == "incorrect"
    assert calls["n"] == 2


def test_judge_local_falls_back_to_erroneous_after_exhausting_retries(monkeypatch):
    calls = {"n": 0}

    def fake_ask_model(*a, **k):
        calls["n"] += 1
        return "garbage nonsense"

    monkeypatch.setattr(judge_mod, "ask_model", fake_ask_model)
    result = judge_mod.judge_local("model", "q", "truth", "answer", retries=1)
    assert result == "erroneous"
    # initial attempt + 1 retry = 2 calls
    assert calls["n"] == 2


def test_judge_local_propagates_non_value_errors(monkeypatch):
    def fake_ask_model(*a, **k):
        raise ConnectionError("backend unreachable")

    monkeypatch.setattr(judge_mod, "ask_model", fake_ask_model)
    with pytest.raises(ConnectionError):
        judge_mod.judge_local("model", "q", "truth", "answer", retries=1)


# ---------------------------------------------------------------------------
# Incremental checkpointing
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_inputs(monkeypatch):
    monkeypatch.setattr(rc, "ask_model", lambda *a, **k: "RAW_DRAFT_TEXT")
    monkeypatch.setattr(rc, "auto_proposed_answers", lambda *a, **k: "PROPOSED")
    monkeypatch.setattr(rc, "decompose_answer",
                        lambda answer, model, **k: ["claim one", "claim two"])
    monkeypatch.setattr(rc, "score_claim_sycophancy", lambda *a, **k: 0.9)
    monkeypatch.setattr(rc, "sycophancy_risk_score", lambda *a, **k: 0.5)
    monkeypatch.setattr(rc, "judge_local", lambda *a, **k: "correct")
    monkeypatch.setattr(rc, "judge_claim_support", lambda *a, **k: True)


def test_calibration_collect_checkpoint_line_count(risk_inputs, tmp_path):
    ckpt = tmp_path / "calib.partial.jsonl"
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple", "ethos"), oracle_truth=True)
    items = [ITEM, {"question": "What is Y?", "answer": "REFERENCE_ANSWER_2"}]

    _, _, _, _, _, records = rc.calibration_collect(
        cfg, items, "scorer", claim_alpha=0.05, checkpoint_path=str(ckpt)
    )

    assert ckpt.exists()
    lines = ckpt.read_text().splitlines()
    assert len(lines) == len(records)
    for line in lines:
        json.loads(line)  # each line is valid JSON


def test_calibration_collect_no_checkpoint_when_path_none(risk_inputs, tmp_path):
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)
    # Should not raise / attempt to write anything when checkpoint_path is None.
    rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05)
    assert list(tmp_path.iterdir()) == []


def test_rerun_truncates_stale_checkpoint(risk_inputs, tmp_path):
    # A rerun with the same --out must not accumulate records from a previous
    # (e.g. crashed) run: main() resets the checkpoint before any work.
    ckpt = tmp_path / "rerun.partial.jsonl"
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)

    def run_once():
        # mirror main(): truncate any stale checkpoint, then run the pipeline
        rc._reset_checkpoint(str(ckpt))
        return rc.calibration_collect(
            cfg, [ITEM], "scorer", claim_alpha=0.05, checkpoint_path=str(ckpt)
        )[-1]

    run_once()  # first run leaves a checkpoint behind
    records = run_once()  # rerun with the same path

    lines = ckpt.read_text().splitlines()
    assert len(lines) == len(records)  # only the second run's rows, no carryover


def test_test_apply_checkpoint_line_count(risk_inputs, tmp_path):
    ckpt = tmp_path / "test.partial.jsonl"
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple", "ethos"), oracle_truth=True)
    fit = ThresholdFitResult(alpha=0.1, tau_global=0.9)
    items = [ITEM, {"question": "What is Y?", "answer": "REFERENCE_ANSWER_2"}]

    df = rc.test_apply(
        cfg, items, "scorer", fit, enable_rewrite=False, claim_threshold=0.5,
        checkpoint_path=str(ckpt),
    )

    assert ckpt.exists()
    lines = ckpt.read_text().splitlines()
    assert len(lines) == len(df)
    for line in lines:
        json.loads(line)


def test_mode_both_uses_separate_phase_checkpoints(risk_inputs, tmp_path):
    # Calibration records and test rows have different schemas and colliding
    # 0-based idx ranges, so mode=both must write each phase to its own file
    # (a shared file would silently mix the two and corrupt crash-salvage).
    calib_ckpt = tmp_path / "out.json.calib.partial.jsonl"
    test_ckpt = tmp_path / "out.json.test.partial.jsonl"
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",), oracle_truth=True)
    fit = ThresholdFitResult(alpha=0.1, tau_global=0.9)

    # mirror main() in mode=both: reset both, calibrate to one, test to the other
    rc._reset_checkpoint(str(calib_ckpt))
    rc._reset_checkpoint(str(test_ckpt))
    records = rc.calibration_collect(
        cfg, [ITEM], "scorer", claim_alpha=0.05, checkpoint_path=str(calib_ckpt)
    )[-1]
    df = rc.test_apply(
        cfg, [ITEM], "scorer", fit, enable_rewrite=False, claim_threshold=0.5,
        checkpoint_path=str(test_ckpt),
    )

    calib_lines = [json.loads(l) for l in calib_ckpt.read_text().splitlines()]
    test_lines = [json.loads(l) for l in test_ckpt.read_text().splitlines()]
    assert len(calib_lines) == len(records)
    assert len(test_lines) == len(df)
    # each file holds only its own phase's schema — no cross-contamination
    assert all("tau_claim" in r and "sycophancy" not in r for r in calib_lines)
    assert all("sycophancy" in r and "tau_claim" not in r for r in test_lines)
