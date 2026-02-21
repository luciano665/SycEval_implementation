#!/bin/bash
#SBATCH --job-name=syceval_nvidia
#SBATCH --output=logs/output_nvidia_%j.txt
#SBATCH --error=logs/error_nvidia_%j.txt
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

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student for baseline
python distill_eval.py \
  --teacher_model ./models/Nemotron-Flash-3B \
  --student_model ./models/Nemotron-Flash-3B \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --out results/final_run_v1/nvidia_baseline.json \
  --backend hf
