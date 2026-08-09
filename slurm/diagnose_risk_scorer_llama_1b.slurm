#!/bin/bash
#SBATCH --job-name=diag_risk_llama_1b
#SBATCH --output=logs/output_diag_risk_llama_1b_%j.txt
#SBATCH --error=logs/error_diag_risk_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_7day

# Diagnostic, not an experiment: pulls a handful of fresh items through the
# real pipeline and captures the risk scorer's RAW reasoning text (the real
# suite only ever saved the parsed number). Answers: is the ceiling effect
# found in analyze_score_separation.py a prompt problem (scorer pattern-
# matches "rebuttal present = risky" without real reasoning) or a
# capability problem (scorer reasons sensibly but still lands high)?
# Small N, fast -- not a full calibration/test run.

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

srun python diagnose_risk_scorer.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain healthsearch \
  --oracle_truth \
  --n_items 15 \
  --max_examples 5 \
  > results/smoke_test/diagnose_risk_scorer_llama_1b_output.txt

echo "--- DONE. Output saved to results/smoke_test/diagnose_risk_scorer_llama_1b_output.txt ---"
