#!/bin/bash
#SBATCH --job-name=v5_phi_2_skeptic
#SBATCH --output=logs/output_v5_phi_2_%j.txt
#SBATCH --error=logs/error_v5_phi_2_%j.txt
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

mkdir -p results/final_experiment_v5

echo "--- Phi 2 V5 Conformal SKEPTIC (N=300, Alpha=0.05, Group-Thresholds) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.33 \
  --max_items 300 \
  --enable_rewrite \
  --use_group_thresholds \
  --thresholds_out results/final_experiment_v5/thresholds_phi_2_v5_skeptic.json \
  --out results/final_experiment_v5/run_conformal_phi_2_v5_skeptic.json
