#!/bin/bash
#SBATCH --job-name=v4_phi_1.5_baseline
#SBATCH --output=logs/output_v4_phi_1.5_baseline_%j.txt
#SBATCH --error=logs/error_v4_phi_1.5_baseline_%j.txt
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

echo "--- Phi 1.5 V4 Baseline (N=300) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model /shared/models/phi-1_5 \
  --judge_model /shared/models/qwen2.5-7b-instruct \
  --rebuttal_model /shared/models/qwen2.5-7b-instruct \
  --max_items 300 \
  --medquad_only \
  --out results/final_experiment_v4/run_baseline_phi_1.5_v4.json
