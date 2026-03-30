#!/bin/bash
#SBATCH --job-name=v6_gemma_4b_baseline
#SBATCH --output=logs/output_v6_gemma_4b_baseline_%j.txt
#SBATCH --error=logs/error_v6_gemma_4b_baseline_%j.txt
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

echo "--- Gemma 4B V5 Baseline (N=1000, MedQuad Only) ---"

srun python run_eval.py \
  --backend hf \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --domain healthsearch --out results/healthsearch_v6_1000/run_baseline_gemma_4b_v5.json
