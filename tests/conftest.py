import sys
from pathlib import Path

# Repo root on sys.path so `from config import EvalConfig` and
# `from conformal_v2 import run_conformal_v2` resolve exactly as they do
# when running the pipeline from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
