#!/bin/bash
#SBATCH --job-name=v4_phi_1.5_conformal
#SBATCH --output=logs/output_v4_phi_1.5_%j.txt
#SBATCH --error=logs/error_v4_phi_1.5_%j.txt
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

echo "--- Phi 1.5 V4 Conformal (N=300) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/phi-1_5 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.1 \
  --calib_frac 0.33 \
  --max_items 300 \
  --enable_rewrite \
  --thresholds_out results/final_experiment_v4/thresholds_phi_1.5_v4.json \
  --out results/final_experiment_v4/run_conformal_phi_1.5_v4.json
