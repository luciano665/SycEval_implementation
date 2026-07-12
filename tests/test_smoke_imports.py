def test_pipeline_imports():
    """The CCR runner must be importable without torch/transformers/ollama."""
    from conformal_v2 import run_conformal_v2  # noqa: F401
    from conformal_v2.conformal_thresholds import ThresholdFitResult  # noqa: F401
    from config import EvalConfig  # noqa: F401
