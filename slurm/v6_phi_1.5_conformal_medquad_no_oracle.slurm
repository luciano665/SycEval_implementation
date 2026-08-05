#!/bin/bash
#SBATCH --job-name=v6_phi_1.5_mq_no_oracle
#SBATCH --output=logs/output_v6_phi_1.5_mq_no_oracle_%j.txt
#SBATCH --error=logs/error_v6_phi_1.5_mq_no_oracle_%j.txt
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

echo "--- Phi 1.5 V6 Conformal DEPLOYABLE (N=1000, MedQuad, Alpha=0.05, Group-Thresholds, no_oracle_truth) ---"

srun python -m conformal_v2.run_conformal_v2 \
  --tested_model models/phi-1_5 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.25 \
  --max_items 1000 \
  --enable_rewrite \
  --use_group_thresholds \
  --no_oracle_truth \
  --thresholds_out results/medDataset_v6_1000/thresholds_phi_1.5_no_oracle_v6.json \
  --domain medquad --out results/medDataset_v6_1000/run_conformal_phi_1.5_no_oracle_v6.json
