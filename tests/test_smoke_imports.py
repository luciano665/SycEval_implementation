def test_pipeline_imports():
    """The CCR runner must be importable without torch/transformers/ollama."""
    from conformal_v2 import run_conformal_v2  # noqa: F401
    from conformal_v2.conformal_thresholds import ThresholdFitResult  # noqa: F401
    from config import EvalConfig  # noqa: F401


def test_metrics_imports():
    """metrics.py imports cleanly and its source references no private pandas internals.

    The source-level check guards against reintroducing imports like
    `from pandas.core.missing import F`, which may happen to import on the
    installed pandas version but break on others.
    """
    import metrics  # noqa: F401

    with open(metrics.__file__) as f:
        source = f.read()
    assert "pandas.core" not in source


def test_evaluation_analysis_is_valid_python():
    """evaluation_analysis.py must be syntactically valid (no unresolved merge conflicts).

    Not imported directly because it is a script that may execute analysis
    logic at import time.
    """
    import ast
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "evaluation_analysis.py")
    with open(path) as f:
        source = f.read()
    ast.parse(source)
