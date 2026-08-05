#!/bin/bash
#SBATCH --job-name=v6_gemma_1b_mq_oracle
#SBATCH --output=logs/output_v6_gemma_1b_mq_oracle_%j.txt
#SBATCH --error=logs/error_v6_gemma_1b_mq_oracle_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_7day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/medDataset_v6_1000

echo "--- Gemma 1B V6 Conformal ORACLE (N=1000, MedQuad, Alpha=0.05, Group-Thresholds, oracle_truth) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.25 \
  --max_items 1000 \
  --enable_rewrite \
  --use_group_thresholds \
  --oracle_truth \
  --thresholds_out results/medDataset_v6_1000/thresholds_gemma_1b_oracle_v6.json \
  --domain medquad --out results/medDataset_v6_1000/run_conformal_gemma_1b_oracle_v6.json
