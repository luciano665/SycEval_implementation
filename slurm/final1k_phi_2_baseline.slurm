#!/bin/bash
#SBATCH --job-name=final1k_phi_2_base
#SBATCH --output=logs/output_final1k_phi_2_base_%j.txt
#SBATCH --error=logs/error_final1k_phi_2_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpu_2day

# Final experiment (N=1000): BASELINE for phi_2 (MedQuad).
# Same protocol as v9n/v10, scaled up: N=1000 instead of 300, chosen for
# a more substantial final dataset. calib_frac=0.2 -> 200 calib / 800 test
# (not the 50/50 split used at N=300) -- justified by
# analyze_calib_size_projection.py: calibration's achievable benefit
# saturates by ~200 items (confirmed against the real v9 data: 200 vs
# 300 vs 400 calib items gave near-identical projected alpha_min for the
# one borderline model affected), so the extra items are better spent on
# test-set precision.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/final_n1000_medquad

echo "--- final N=1000 BASELINE phi_2 (MedQuad, Qwen judge) ---"
srun python run_eval.py \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 1000 \
  --domain medquad \
  --out results/final_n1000_medquad/run_baseline_phi_2.json
echo "--- baseline phi_2 exited 0 ---"
