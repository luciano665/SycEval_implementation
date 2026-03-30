#!/bin/bash
#SBATCH --job-name=v6_gemma_4b_adversarial
#SBATCH --output=logs/output_v6_gemma_4b_adv_%j.txt
#SBATCH --error=logs/error_v6_gemma_4b_adv_%j.txt
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

mkdir -p results/healthsearch_v6_1000

echo "--- Gemma 4B V6 ADVERSARIAL OVERSIGHT (N=1000, Alpha=0.05, Group-Thresholds) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.25 \
  --max_items 1000 \
  --enable_rewrite \
  --use_group_thresholds \
  --thresholds_out results/healthsearch_v6_1000/thresholds_gemma_4b_v6.json \
  --domain healthsearch --out results/healthsearch_v6_1000/run_conformal_gemma_4b_v6.json
