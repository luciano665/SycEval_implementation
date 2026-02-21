#!/bin/bash
#SBATCH --job-name=qwen_1.5B_base
#SBATCH --output=logs/output_qwen_1.5B_base_%j.txt
#SBATCH --error=logs/error_qwen_1.5B_base_%j.txt
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
mkdir -p results/ministral_judge_v1

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student = Qwen 1.5B
# Judge: Ministral-8B-Instruct-2410
echo "--- Qwen 1.5B Baseline ---"

python distill_eval.py \
  --teacher_model ./models/Qwen2.5-1.5B-Instruct \
  --student_model ./models/Qwen2.5-1.5B-Instruct \
  --rebuttal_model ./models/Ministral-8B-Instruct-2410 \
  --judge_model ./models/Ministral-8B-Instruct-2410 \
  --max_items 1000 \
  --out results/ministral_judge_v1/qwen_1.5B_baseline.json \
  --backend hf
