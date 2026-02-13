#!/bin/bash
#SBATCH --job-name=gemma_1B_base
#SBATCH --output=logs/output_gemma_1B_base_%j.txt
#SBATCH --error=logs/error_gemma_1B_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

# Create results directory if it doesn't exist
mkdir -p results/ministral_judge_v1

# Run Distillation Evaluation (V3 Baseline)
# Teacher = Student = Gemma-1B
# Judge: Ministral-8B-Instruct-2410
echo "--- Gemma 1B Baseline ---"

python distill_eval.py \
  --teacher_model ./models/gemma-3-1b-it \
  --student_model ./models/gemma-3-1b-it \
  --rebuttal_model ./models/Ministral-8B-Instruct-2410 \
  --judge_model ./models/Ministral-8B-Instruct-2410 \
  --max_items 1000 \
  --out results/ministral_judge_v1/gemma_1B_baseline.json \
  --backend hf
