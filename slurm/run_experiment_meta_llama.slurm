#!/bin/bash
#SBATCH --job-name=syceval_llama
#SBATCH --output=logs/output_llama_%j.txt
#SBATCH --error=logs/error_llama_%j.txt
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
mkdir -p results/ministral_judge_v1

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student for baseline
# Judge: Ministral-8B-Instruct-2410
echo "--- Llama Baseline ---"

python distill_eval.py \
  --teacher_model ./models/Llama-3.2-3B-Instruct \
  --student_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Ministral-8B-Instruct-2410 \
  --judge_model ./models/Ministral-8B-Instruct-2410 \
  --max_items 1000 \
  --out results/ministral_judge_v1/llama_baseline.json \
  --backend hf
