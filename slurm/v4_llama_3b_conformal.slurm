#!/bin/bash
#SBATCH --job-name=v4_llama_3b_conformal
#SBATCH --output=logs/output_v4_llama_3b_%j.txt
#SBATCH --error=logs/error_v4_llama_3b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/final_experiment_v4

echo "--- Llama 3B V4 Conformal (N=300) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model /shared/models/llama-3.2-3b-instruct \
  --judge_model /shared/models/qwen2.5-7b-instruct \
  --rebuttal_model /shared/models/qwen2.5-7b-instruct \
  --alpha 0.1 \
  --calib_frac 0.33 \
  --max_items 300 \
  --medquad_only \
  --enable_rewrite \
  --out results/final_experiment_v4/run_conformal_llama_3b_v4.json
