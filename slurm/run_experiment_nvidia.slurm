#!/bin/bash
#SBATCH --job-name=syceval_nvidia
#SBATCH --output=logs/output_nvidia_%j.txt
#SBATCH --error=logs/error_nvidia_%j.txt
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

srun --unbuffered python -u distill_eval.py \
  --backend hf \
  --teacher_model ./models/Nemotron-Flash-3B \
  --student_model ./models/Nemotron-Flash-1B \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --max_items 100 \
  --out nvidia_distill_results_100.json
