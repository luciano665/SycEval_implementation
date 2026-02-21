#!/bin/bash
#SBATCH --job-name=b_gemma_1b_base
#SBATCH --output=logs/b_gemma_1b_base_%j.txt
#SBATCH --error=logs/err_b_gemma_1b_base_%j.txt
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
mkdir -p results/final_run_v1

# Match Suite B Baseline Flow (distill_eval.py student=teacher)
python distill_eval.py \
  --teacher_model ./models/gemma-3-1b-it \
  --student_model ./models/gemma-3-1b-it \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --backend hf \
  --out results/final_experiment_v1/suite_b_gemma_1b_baseline.json
