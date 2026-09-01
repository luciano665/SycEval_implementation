import pandas as pd
import pytest

from conformal_v2 import run_conformal_v2 as rc
from conformal_v2.conformal_thresholds import ThresholdFitResult, fit_global_threshold
from config import EvalConfig

ITEM = {"question": "What is X?", "answer": "REFERENCE_ANSWER"}


def test_fit_global_threshold_sentinel_unchanged():
    # alpha unattainable -> sentinel -1.0 (existing behavior, kept)
    assert fit_global_threshold([0.1, 0.2], [1, 1], alpha=0.05) == -1.0


def test_fit_thresholds_sets_calibration_failed():
    fit = rc.fit_thresholds(scores=[0.1, 0.2], bad=[1, 1], groups=["g", "g"],
                            alpha=0.05, use_group_thresholds=False)
    assert fit.calibration_failed is True
    assert fit.tau_global == -1.0


def test_fit_thresholds_success_not_flagged():
    # 100 accepted, 0 bad -> Wilson UB ~0.037 <= 0.05 -> valid tau
    scores = [i / 1000 for i in range(100)]
    fit = rc.fit_thresholds(scores=scores, bad=[0] * 100, groups=["g"] * 100,
                            alpha=0.05, use_group_thresholds=False)
    assert fit.calibration_failed is False
    assert fit.tau_global == max(scores)


def test_threshold_json_roundtrip_with_flags():
    fit = ThresholdFitResult(alpha=0.05, tau_global=-1.0, tau_by_group=None,
                             calibration_failed=True)
    d = rc.threshold_to_json(fit, tau_claim=0.3, claim_alpha=0.05,
                             tau_claim_fallback=True)
    assert d["calibration_failed"] is True
    assert d["tau_claim_fallback"] is True
    assert "oracle_truth" not in d  # removed in conformal_v9 (deployable-only)
    fit2, tau_claim2, claim_alpha2, fallback2 = rc.thresholds_from_json(d)
    assert fit2.calibration_failed is True
    assert fit2.tau_global == -1.0
    assert tau_claim2 == 0.3 and claim_alpha2 == 0.05
    assert fallback2 is True


def test_thresholds_from_json_legacy_files():
    # Old v6 files lack the new keys: infer failure from tau_global<0,
    # leave the unknowables as None.
    legacy = {"alpha": 0.05, "tau_global": -1.0, "tau_by_group": None,
              "tau_claim": 0.3, "claim_alpha": 0.05}
    fit, tau_claim, claim_alpha, fallback = rc.thresholds_from_json(legacy)
    assert fit.calibration_failed is True
    assert fallback is None
    # Also legacy for threshold_method/xi_certified/alpha_min: every run
    # before exact_crc existed used Wilson, and the diagnostic fields are
    # unknowable without recomputing from the (possibly discarded) checkpoint.
    assert fit.threshold_method == "wilson"
    assert fit.xi_certified is None
    assert fit.alpha_min is None


def test_fit_thresholds_default_method_is_wilson():
    # No threshold_method passed -> identical to pre-exact-CRC behavior.
    scores = [i / 1000 for i in range(100)]
    fit = rc.fit_thresholds(scores=scores, bad=[0] * 100, groups=["g"] * 100,
                            alpha=0.05, use_group_thresholds=False)
    assert fit.threshold_method == "wilson"
    assert fit.tau_global == max(scores)


def test_fit_thresholds_exact_crc_rescues_small_n_case():
    # Same 19-clean-point case from test_selective_crc.py: Wilson fails,
    # exact_crc certifies with the exact same data, no more items needed.
    scores = [i / 100 for i in range(19)]
    fit_wilson = rc.fit_thresholds(scores=scores, bad=[0] * 19, groups=["g"] * 19,
                                   alpha=0.05, use_group_thresholds=False,
                                   threshold_method="wilson")
    fit_exact = rc.fit_thresholds(scores=scores, bad=[0] * 19, groups=["g"] * 19,
                                  alpha=0.05, use_group_thresholds=False,
                                  threshold_method="exact_crc")
    assert fit_wilson.calibration_failed is True
    assert fit_exact.calibration_failed is False
    assert fit_exact.threshold_method == "exact_crc"
    assert fit_exact.tau_global == max(scores)
    assert fit_exact.xi_certified == pytest.approx(1.0)


def test_fit_thresholds_rejects_unknown_method():
    with pytest.raises(ValueError):
        rc.fit_thresholds(scores=[0.1], bad=[0], groups=["g"],
                          alpha=0.05, use_group_thresholds=False,
                          threshold_method="not_a_real_method")


def test_threshold_json_roundtrip_carries_method_and_diagnostics():
    fit = ThresholdFitResult(alpha=0.10, tau_global=0.42, tau_by_group=None,
                             calibration_failed=False,
                             threshold_method="exact_crc",
                             xi_certified=0.87, alpha_min=0.03)
    d = rc.threshold_to_json(fit, tau_claim=0.3, claim_alpha=0.05)
    assert d["threshold_method"] == "exact_crc"
    assert d["xi_certified"] == pytest.approx(0.87)
    assert d["alpha_min"] == pytest.approx(0.03)

    fit2, *_ = rc.thresholds_from_json(d)
    assert fit2.threshold_method == "exact_crc"
    assert fit2.xi_certified == pytest.approx(0.87)
    assert fit2.alpha_min == pytest.approx(0.03)


def test_rewrite_rate_summary():
    df = pd.DataFrame({"where": ["in-context", "preemptive"],
                       "rewrite_triggered": [True, False]})
    r = rc.rewrite_rate_summary(df)
    assert r == {"overall": 0.5, "in_context": 1.0, "preemptive": 0.0}


def test_rewrite_rate_summary_empty():
    assert rc.rewrite_rate_summary(pd.DataFrame()) == {
        "overall": 0.0, "in_context": 0.0, "preemptive": 0.0}


def test_oracle_risk_scorer_calib_passes_truth_to_scorer(monkeypatch):
    # Default (flag off): risk scorer must be called with truth=None, same
    # as every deployable-only run before this flag existed.
    seen_truth = []

    def fake_risk_score(*a, **k):
        seen_truth.append(k.get("truth"))
        return 0.5

    monkeypatch.setattr(rc, "ask_model", lambda *a, **k: "RAW_DRAFT_TEXT")
    monkeypatch.setattr(rc, "auto_proposed_answers", lambda *a, **k: "PROPOSED")
    monkeypatch.setattr(rc, "decompose_answer", lambda answer, model, **k: ["c1"])
    monkeypatch.setattr(rc, "score_claim_sycophancy", lambda *a, **k: 0.9)
    monkeypatch.setattr(rc, "sycophancy_risk_score", fake_risk_score)
    monkeypatch.setattr(rc, "judge_local", lambda *a, **k: "correct")
    monkeypatch.setattr(rc, "judge_claim_support", lambda *a, **k: False)
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",))

    rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05,
                           oracle_risk_scorer_calib=False)
    assert seen_truth == [None, None]  # in-context + preemptive, both simple strength

    seen_truth.clear()
    rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05,
                           oracle_risk_scorer_calib=True)
    assert seen_truth == [ITEM["answer"], ITEM["answer"]]


def test_tau_claim_fallback_flag(monkeypatch):
    # All claims unsupported -> no valid claim threshold -> fallback 0.3 + flag.
    monkeypatch.setattr(rc, "ask_model", lambda *a, **k: "RAW_DRAFT_TEXT")
    monkeypatch.setattr(rc, "auto_proposed_answers", lambda *a, **k: "PROPOSED")
    monkeypatch.setattr(rc, "decompose_answer", lambda answer, model, **k: ["c1", "c2"])
    monkeypatch.setattr(rc, "score_claim_sycophancy", lambda *a, **k: 0.9)
    monkeypatch.setattr(rc, "sycophancy_risk_score", lambda *a, **k: 0.5)
    monkeypatch.setattr(rc, "judge_local", lambda *a, **k: "correct")
    monkeypatch.setattr(rc, "judge_claim_support", lambda *a, **k: False)
    cfg = EvalConfig(backend="hf", rebuttal_strengths=("simple",))
    tau_claim, tau_claim_fallback, scores, bad, groups, records = \
        rc.calibration_collect(cfg, [ITEM], "scorer", claim_alpha=0.05)
    assert tau_claim == 0.3
    assert tau_claim_fallback is True
