#!/bin/bash
#SBATCH --job-name=v10_phi_2_base
#SBATCH --output=logs/output_v10_phi_2_base_%j.txt
#SBATCH --error=logs/error_v10_phi_2_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# FALLBACK ONLY -- do not run this by default. See v10_llama_1b_baseline.slurm
# for the full explanation. Run only if
# results/v9_night_medquad/run_baseline_phi_2.json is unavailable.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v10_exact_crc_medquad

echo "--- v10 BASELINE (fallback) phi_2 (N=300, MedQuad, Qwen judge) ---"
srun python run_eval.py \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v10_exact_crc_medquad/run_baseline_phi_2.json
echo "--- baseline phi_2 exited 0 ---"
