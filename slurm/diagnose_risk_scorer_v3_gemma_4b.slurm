#!/bin/bash
#SBATCH --job-name=diag_risk_v3_gemma_4b
#SBATCH --output=logs/output_diag_risk_v3_gemma_4b_%j.txt
#SBATCH --error=logs/error_diag_risk_v3_gemma_4b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_7day

# Generalization check: same OLD vs v2 vs v3 comparison, but on Gemma-4B --
# the other model with a persistent (data-proof) calibration ceiling --
# instead of Llama-1B, to see whether the prompt fix's effect holds up on
# a different model, not just the one it was tuned against.

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

srun python diagnose_risk_scorer_v3.py \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain healthsearch \
  --oracle_truth \
  --n_items 15 \
  > results/smoke_test/diagnose_risk_scorer_v3_gemma_4b_output.txt

echo "--- DONE. Output saved to results/smoke_test/diagnose_risk_scorer_v3_gemma_4b_output.txt ---"
