#!/bin/bash
#SBATCH --job-name=v6_phi_2
#SBATCH --output=logs/output_v6_phi_2_%j.txt
#SBATCH --error=logs/error_v6_phi_2_%j.txt
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

mkdir -p results/medDataset_v6_1000

echo "--- Phi 2 V5 Conformal SKEPTIC (N=1000, Alpha=0.05, Group-Thresholds) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.25 \
  --max_items 1000 \
  --enable_rewrite \
  --use_group_thresholds \
  --thresholds_out results/medDataset_v6_1000/thresholds_phi_2_v6.json \
  --out results/medDataset_v6_1000/run_conformal_phi_2_v6.json
