#!/bin/bash
#SBATCH --job-name=v9j_gemma_1b_conf
#SBATCH --output=logs/output_v9j_gemma_1b_conf_%j.txt
#SBATCH --error=logs/error_v9j_gemma_1b_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# Independent-judge suite -- CONFORMAL (CCR) for gemma_1b (MedQuad, N=300).
# Same structure as v9n_gemma_1b_conformal.slurm (deployable-only, leak-free
# rewrite, folding-only labels, global thresholds, alpha=0.10), but
# judge_model=Mistral-7B-Instruct-v0.3 (independent of rebuttal/scorer).
# risk_scorer_model is EXPLICIT (Qwen) -- it must not be left to default,
# or it silently inherits judge_model=Mistral via the
# `args.risk_scorer_model.strip() or cfg.judge_model` fallback in
# run_conformal_v2.py.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_judge_split_medquad

echo "--- referee-split CONFORMAL gemma_1b (N=300, alpha=0.10, Mistral judge) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Mistral-7B-Instruct-v0.3 \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 300 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/v9_judge_split_medquad/thresholds_gemma_1b.json \
  --out results/v9_judge_split_medquad/run_conformal_gemma_1b.json
echo "--- conformal gemma_1b exited 0 ---"
