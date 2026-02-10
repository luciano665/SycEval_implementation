#!/bin/bash
#SBATCH --job-name=syceval_gemma
#SBATCH --output=logs/output_gemma_%j.txt
#SBATCH --error=logs/error_gemma_%j.txt
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

# Create results directory if it doesn't exist
mkdir -p results/final_run_v1

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student for baseline
# Judge: Mistral-7B-Instruct-v0.2
echo "--- Gemma Baseline ---"

python distill_eval.py \
  --teacher_model ./models/gemma-3-4b-it \
  --student_model ./models/gemma-3-4b-it \
  --rebuttal_model ./models/Mistral-7B-Instruct-v0.2 \
  --judge_model ./models/Mistral-7B-Instruct-v0.2 \
  --max_items 1000 \
  --out results/final_run_v1/gemma_baseline.json \
  --backend hf
