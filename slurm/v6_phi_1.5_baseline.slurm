#!/bin/bash
#SBATCH --job-name=v6_phi_1.5_baseline
#SBATCH --output=logs/output_v6_phi_1.5_baseline_%j.txt
#SBATCH --error=logs/error_v6_phi_1.5_baseline_%j.txt
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

echo "--- Phi 1.5 V5 Baseline (N=300, MedQuad Only) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model models/phi-1_5 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 300 \
  --out results/final_experiment_v6/run_baseline_phi_1.5_v5.json
