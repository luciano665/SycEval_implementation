#!/bin/bash
#SBATCH --job-name=v5_llama_3b_baseline
#SBATCH --output=logs/output_v5_llama_3b_baseline_%j.txt
#SBATCH --error=logs/error_v5_llama_3b_baseline_%j.txt
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

echo "--- Llama 3B V5 Baseline (N=300, MedQuad Only) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 300 \
  --out results/final_experiment_v5/run_baseline_llama_3b_v5.json
