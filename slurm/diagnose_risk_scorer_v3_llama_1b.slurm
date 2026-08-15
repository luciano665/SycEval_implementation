#!/bin/bash
#SBATCH --job-name=diag_risk_v3_llama_1b
#SBATCH --output=logs/output_diag_risk_v3_llama_1b_%j.txt
#SBATCH --error=logs/error_diag_risk_v3_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_7day

# Round 2 comparison: OLD vs v2 vs v3 prompt, same model as the v2 test
# (Llama-1B) to see whether sharpening the (b)/(c) distinction actually
# fixes the "genuine folds miscategorized as declined" issue v2 introduced.

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

srun python diagnose_risk_scorer_v3.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain healthsearch \
  --oracle_truth \
  --n_items 15 \
  > results/smoke_test/diagnose_risk_scorer_v3_llama_1b_output.txt

echo "--- DONE. Output saved to results/smoke_test/diagnose_risk_scorer_v3_llama_1b_output.txt ---"
