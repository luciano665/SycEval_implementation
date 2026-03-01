#!/bin/bash
#SBATCH --job-name=syceval_gemma
#SBATCH --output=logs/output_gemma_%j.txt
#SBATCH --error=logs/error_gemma_%j.txt
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
mkdir -p results/final_experiment_v3

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student for baseline
# Judge: Qwen2.5-7B-Instruct
echo "--- Gemma 4B Baseline v3 ---"

python distill_eval.py \
  --teacher_model ./models/gemma-3-4b-it \
  --student_model ./models/gemma-3-4b-it \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --out results/final_experiment_v3/suite_b_gemma_4b_baseline_v3.json \
  --backend hf
