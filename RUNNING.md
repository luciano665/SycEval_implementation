# Local & HF Evaluation Guide

This guide explains how to run `run_eval.py` on a local CSV (`data_loader.load_data_local`) or directly from the Hugging Face dataset (`data_loader.load_data`). It also covers how to switch between the Ollama and Hugging Face backends defined in `config.py`.

## 1. Prerequisites
- Python 3.10+
- Project dependencies:
  ```bash
  cd /Users/lucianom/Research_realiability_LLMs/SycEval
  uv pip install -r requirements.txt
  ```
- Optional local CSV: place `medDataset_processed.csv` under `data/` (the default location expected by `load_data_local`).

## 2. Choosing the Data Source
`run_eval.py` currently calls `load_data_local`:
```python
data = load_data_local(
    n=cfg.max_items,
    seed=seed,
    csv_path="data/medDataset_processed.csv"
)
```

| Loader | Function | When to use | Notes |
|--------|----------|-------------|-------|
| Local CSV | `load_data_local` |  `data/medDataset_processed.csv` | Samples locally, no network calls |
| Hugging Face | `load_data` |  HF-hosted `keivalya/MedQuad-MedicalQnADataset` | Requires internet, removes need for local CSV |

To switch loaders, edit `run_eval.py` (inside `run_medquad`) and replace the line above with `data = load_data(...)`.

## 3. Backend Selection (`config.py`)
`EvalConfig.backend` defaults to `ollama`, and the same value propagates through every call to `ask_model`. You can override it:
```bash
python run_eval.py --backend ollama ...
python run_eval.py --backend hf ...
```

### 3.1 Ollama Backend
1. Start the Ollama server:
   ```bash
   ollama serve
   ```
2. Run the evaluation (using default Ollama model tags):
   ```bash
   python run_eval.py \
     --backend ollama \
     --tested_model llama3.2:3b \
     --rebuttal_model gemma3:1b \
     --judge_model llama3:8b \
     --max_items 4 \
     --temperature 0.0 \
     --out medquad_eval.json
   ```

### 3.2 Hugging Face Backend
Use HF model IDs; the script will download weights on first run.
```bash
python run_eval.py \
  --backend hf \
  --tested_model mistralai/Mistral-7B-Instruct-v0.2 \
  --rebuttal_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --judge_model Qwen/Qwen2.5-7B-Instruct \
  --max_items 4 \
  --temperature 0.0 \
  --out medquad_eval.json
```

## 4. Understanding `run_eval.py`
- Parses CLI flags, instantiates `EvalConfig`, and passes the backend to every `ask_model` call.
- Uses:
  - `initial_answer` → first model response + judge label
  - `in_context_chain` / `preemptive_chain` → apply rebuttals
  - `judge_local` → evaluates answers using the configured backend
  - `auto_proposed_answers` → generates rebuttal claims (same backend)
- Outputs summary stats plus a JSON file with per-question traces.
