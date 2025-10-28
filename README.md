# SycEval: Sycophancy Evaluation Framework Implementation


SycEval is a framework for evaluating sycophancy in large language models using medical question-answering tasks. The system tests how models respond to user rebuttals and measures both progressive sycophancy (changing incorrect answers to correct ones) and regressive sycophancy (changing correct answers to incorrect ones).

## Overview

This evaluation framework implements a three-phase approach:
1. **Phase 1**: Get initial model answers to medical questions
2. **Phase 2**: Generate rebuttals using a separate model
3. **Phase 3**: Test model responses to rebuttals in two modes:
   - **In-context**: Show rebuttal after initial answer
   - **Preemptive**: Show rebuttal before asking the question

## Prerequisites

- Python 3.7+
- Ollama installed and running locally
- Required Ollama models pulled (see Setup section)

## Setup

### 0. Create env for dpeendencies
 - I use uv for creating and venv but you can use any and install what is below "requiments"

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Setup Ollama

1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start the Ollama service:
   ```bash
   ollama serve
   ```

3. Pull the required models:
   ```bash
   # Tested model (the model being evaluated)
   ollama pull llama3.2:3b
   
   # Rebuttal model (generates rebuttals)
   ollama pull gemma3:1b
   
   # Judge model (evaluates answer correctness)
   ollama pull llama3:8b
   ```

### 3. Verify Installation

Test that Ollama is working:
```bash
ollama list
```

You should see the three models listed above.

## Usage

### Basic Usage

Run the evaluation with default settings:
```bash
python run_eval.py
```

### Custom Configuration

Run with custom parameters:
```bash
python run_eval.py \
    --max_items 200 \
    --tested_model llama3.2:3b \
    --rebuttal_model gemma3:1b \
    --judge_model llama3:8b \
    --temperature 0.0 \
    --out medquad_eval.jsonl
```

### Parameters

- `--max_items`: Number of MedQuad Q/A pairs to sample (default: 20)
- `--tested_model`: Model to evaluate for sycophancy (default: llama3.2:3b)
- `--rebuttal_model`: Model used to generate rebuttals (default: gemma3:1b)
- `--judge_model`: Model used to judge answer correctness (default: llama3:8b)
- `--temperature`: Temperature for model generation (default: 0.0)
- `--out`: Output file path for results (default: medquad_eval.jsonl)

## Output

The evaluation produces:

1. **Console Output**: Statistical summaries including:
   - Overall sycophancy rates
   - In-context vs preemptive comparison
   - Two-proportion z-test results

2. **JSONL File**: Detailed results with one row per evaluation attempt, containing:
   - Question index
   - Context type (in-context/preemptive)
   - Rebuttal strength
   - Initial and final answer labels
   - Sycophancy classification
   - Original question

## Results in detail

- **Progressive Sycophancy**: Model changes from incorrect to correct answer
- **Regressive Sycophancy**: Model changes from correct to incorrect answer
- **None**: No sycophancy detected

The framework tests four rebuttal strengths:
- `simple`: Basic disagreement
- `ethos`: Authority-based argument
- `justification`: Detailed reasoning
- `citation`: Evidence-based argument

## Troubleshooting

### Common Issues

1. **"Connection refused" error**: Make sure Ollama is running (`ollama serve`)

2. **Model not found**: Ensure all required models are pulled (`ollama pull <model_name>`)

3. **Slow performance**: Consider using smaller models or reducing `--max_items`

4. **Memory issues**: Reduce batch size or use models with fewer parameters

### Getting Help

If you encounter issues:
1. Check that Ollama is running: `ollama list`
2. Verify model availability: `ollama list`
3. Test a simple query: `ollama run llama3.2:3b "Hello"`

## Data Source

The evaluation uses the MedQuad-MedicalQnADataset from Hugging Face, which contains medical question-answer pairs for testing model behavior in a domain-specific context.