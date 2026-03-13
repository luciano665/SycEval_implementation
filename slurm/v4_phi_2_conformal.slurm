#!/bin/bash
#SBATCH --job-name=v4_phi_2_conformal
#SBATCH --output=logs/output_v4_phi_2_%j.txt
#SBATCH --error=logs/error_v4_phi_2_%j.txt
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

mkdir -p results/final_experiment_v4

echo "--- Phi 2 V4 Conformal (N=300) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model /shared/models/phi-2 \
  --judge_model /shared/models/qwen2.5-7b-instruct \
  --rebuttal_model /shared/models/qwen2.5-7b-instruct \
  --alpha 0.1 \
  --calib_frac 0.33 \
  --max_items 300 \
  --enable_rewrite \
  --thresholds_out results/final_experiment_v4/thresholds_phi_2_v4.json \
  --out results/final_experiment_v4/run_conformal_phi_2_v4.json
