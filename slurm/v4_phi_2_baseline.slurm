#!/bin/bash
#SBATCH --job-name=v4_phi_2_baseline
#SBATCH --output=logs/output_v4_phi_2_baseline_%j.txt
#SBATCH --error=logs/error_v4_phi_2_baseline_%j.txt
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

echo "--- Phi 2 V4 Baseline (N=300) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 300 \
  --out results/final_experiment_v4/run_baseline_phi_2_v4.json
