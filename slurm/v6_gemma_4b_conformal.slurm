#!/bin/bash
#SBATCH --job-name=v6_gemma_4b_adversarial
#SBATCH --output=logs/output_v6_gemma_4b_adv_%j.txt
#SBATCH --error=logs/error_v6_gemma_4b_adv_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/final_experiment_v6

echo "--- Gemma 4B V6 ADVERSARIAL OVERSIGHT (N=300, Alpha=0.05, Group-Thresholds) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.33 \
  --max_items 300 \
  --enable_rewrite \
  --use_group_thresholds \
  --thresholds_out results/final_experiment_v6/thresholds_gemma_4b_v6.json \
  --out results/final_experiment_v6/run_conformal_gemma_4b_v6.json
