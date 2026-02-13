#!/bin/bash
#SBATCH --job-name=neutral_qwen_base
#SBATCH --output=logs/neutral_qwen_base_%j.txt
#SBATCH --error=logs/error_neutral_qwen_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Ensure results directory exists
mkdir -p results/neutral_suite_v1

# Run Qwen 3B Baseline with Ministral Judge/Rebuttal
echo "--- Qwen 3B Neutral Baseline ---"
python distill_eval.py \
  --teacher_model ./models/Qwen2.5-3B-Instruct \
  --student_model ./models/Qwen2.5-3B-Instruct \
  --rebuttal_model ./models/Ministral-8B-Instruct-2410 \
  --judge_model ./models/Ministral-8B-Instruct-2410 \
  --max_items 1000 \
  --out results/neutral_suite_v1/qwen_3B_baseline.json \
  --backend hf
