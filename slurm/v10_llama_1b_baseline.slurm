#!/bin/bash
#SBATCH --job-name=v10_llama_1b_base
#SBATCH --output=logs/output_v10_llama_1b_base_%j.txt
#SBATCH --error=logs/error_v10_llama_1b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# FALLBACK ONLY -- do not run this by default. v10 is designed to reuse
# results/v9_night_medquad/run_baseline_llama_1b.json as its baseline
# (identical model/judge/seed/data, and baseline generation never touches
# threshold fitting, so nothing about it changed between v9 and v10).
# Run THIS script only if that v9 file is missing/unavailable -- see
# docs/V10_EXPERIMENT_TRACKER.md's "no v9 baseline available" note.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v10_exact_crc_medquad

echo "--- v10 BASELINE (fallback) llama_1b (N=300, MedQuad, Qwen judge) ---"
srun python run_eval.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v10_exact_crc_medquad/run_baseline_llama_1b.json
echo "--- baseline llama_1b exited 0 ---"
