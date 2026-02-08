#!/bin/bash
#SBATCH --job-name=conformal_llama_3B
#SBATCH --output=logs/output_conformal_llama_3B_%j.txt
#SBATCH --error=logs/error_conformal_llama_3B_%j.txt
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

srun --unbuffered python -m conformal_v2.run_conformal_v2 \
  --backend hf \
  --tested_model ./models/Llama-3.2-3B-Instruct \
  --rebuttal_model ./models/Llama-3.2-3B-Instruct \
  --judge_model ./models/Llama-3.1-8B-Instruct \
  --max_items 100 \
  --mode both \
  --calib_frac 0.5 \
  --alpha 0.1 \
  --enable_rewrite \
  --out results/llama_3B_conformal_results.json
