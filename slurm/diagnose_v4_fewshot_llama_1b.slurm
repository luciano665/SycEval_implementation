#!/bin/bash
#SBATCH --job-name=diag_v4_fewshot
#SBATCH --output=logs/output_diag_v4_fewshot_%j.txt
#SBATCH --error=logs/error_diag_v4_fewshot_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day

# Cheapest possible test of the few-shot idea: does adding 3 hand-written
# worked examples to v3's prompt improve AUC over v3 alone? Live A/B/C
# comparison (OLD vs v3 vs v4-fewshot) on the same N=20 items, one model
# (llama_1b -- our most-tested borderline case). This is a quick gate,
# not a full verification -- if v4-fewshot doesn't show a real AUC jump
# here, not worth pursuing further given the timeline.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME

echo "--- v4-fewshot quick gate: llama_1b (N=20) ---"
srun python diagnose_risk_scorer_v4_fewshot.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain medquad \
  --n_items 20
echo "--- exited 0 ---"
