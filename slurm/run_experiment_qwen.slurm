#!/bin/bash
#SBATCH --job-name=syceval_qwen
#SBATCH --output=logs/output_qwen_%j.txt
#SBATCH --error=logs/error_qwen_%j.txt
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

# Create results directory if it doesn't exist
mkdir -p results/final_run_v1

# Models
# Test: Qwen2.5-1.5B & 3B
# Judge: Qwen2.5-7B-Instruct
echo "Running Qwen Experiment (Baseline)"

# 1. Run Qwen 1.5B Baseline
echo "--- Qwen 1.5B Baseline ---"
python distill_eval.py \
  --teacher_model ./models/Qwen2.5-1.5B-Instruct \
  --student_model ./models/Qwen2.5-1.5B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --out results/final_run_v1/qwen_1.5B_baseline.json \
  --backend hf

# 2. Run Qwen 3B Baseline
echo "--- Qwen 3B Baseline ---"
python distill_eval.py \
  --teacher_model ./models/Qwen2.5-3B-Instruct \
  --student_model ./models/Qwen2.5-3B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --out results/final_run_v1/qwen_3B_baseline.json \
  --backend hf
