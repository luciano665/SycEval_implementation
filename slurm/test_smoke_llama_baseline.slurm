#!/bin/bash
#SBATCH --job-name=smoke_test_baseline
#SBATCH --output=logs/smoke_baseline_%j.txt
#SBATCH --error=logs/smoke_baseline_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Create results directory if it doesn't exist
mkdir -p results/smoke_test_v1

# Run Distillation Evaluation (V3 Baseline) - SMOKE TEST (100 items)
python distill_eval.py \
  --teacher_model ./models/Llama-3.2-3B-Instruct \
  --student_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 100 \
  --out results/smoke_test_v1/llama_3B_baseline.json \
  --backend hf
