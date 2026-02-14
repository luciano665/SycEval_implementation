#!/bin/bash
#SBATCH --job-name=b_phi_2_conf
#SBATCH --output=logs/b_phi_2_conf_%j.txt
#SBATCH --error=logs/err_b_phi_2_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --partition=gpu_2day

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/suite_b_standardized

# Conformal Run (Phi-2)
srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/phi-2 \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --alpha 0.1 \
  --calib_frac 0.5 \
  --max_items 1000 \
  --backend hf \
  --out results/suite_b_standardized/phi_2_conformal.json
