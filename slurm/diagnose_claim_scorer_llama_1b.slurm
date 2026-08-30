#!/bin/bash
#SBATCH --job-name=diag_claim_scorer
#SBATCH --output=logs/output_diag_claim_scorer_%j.txt
#SBATCH --error=logs/error_diag_claim_scorer_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day

# Tests the leading hypothesis for why claim-level calibration
# (tau_claim) fails for all 6 models, including Phi where response-level
# calibration succeeds: score_claim_sycophancy is called with truth=None
# (deployable-only), same blind-judgment problem already measured at the
# response level. Computes real claim-level AUC directly via a fresh
# small live run (N=20 items -> however many individual claims those
# decompose into).
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME

echo "--- claim-level scorer diagnostic: llama_1b (N=20 items) ---"
srun python diagnose_claim_scorer.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain medquad \
  --n_items 20
echo "--- exited 0 ---"
