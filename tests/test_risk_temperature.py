from conformal_v2 import syco_risk


def test_scorer_honors_passed_temperature(monkeypatch):
    captured = {}

    def fake_ask(model, prompt, system=None, temperature=None, backend="hf", **kwargs):
        captured["temperature"] = temperature
        return "Final Score: 0.4"

    monkeypatch.setattr(syco_risk, "ask_model", fake_ask)
    syco_risk.sycophancy_risk_score(
        scorer_model="m", question="q", rebuttal="r",
        initial_answer="a0", draft_answer="d", truth=None,
        backend="hf", temperature=0.0)
    assert captured["temperature"] == 0.0

    syco_risk.sycophancy_risk_score(
        scorer_model="m", question="q", rebuttal="r",
        initial_answer="a0", draft_answer="d", truth=None,
        backend="hf", temperature=0.7)
    assert captured["temperature"] == 0.7


def test_set_global_seed_is_callable_and_seeds_python_random():
    import random

    from models import set_global_seed

    set_global_seed(123)
    a = random.random()
    set_global_seed(123)
    b = random.random()
    assert a == b


def test_set_global_seed_seeds_torch_if_available():
    from models import set_global_seed

    try:
        import torch
    except Exception:
        return  # torch not installed locally; covered on cluster

    set_global_seed(777)
    assert torch.initial_seed() == 777
