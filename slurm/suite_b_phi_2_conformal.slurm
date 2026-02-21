#!/bin/bash
#SBATCH --job-name=b_phi_2_conf
#SBATCH --output=logs/b_phi_2_conf_%j.txt
#SBATCH --error=logs/err_b_phi_2_conf_%j.txt
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

# Exact Suite B Conformal Flow
srun python -m conformal_v2.run_conformal_v2 \
  --tested_model ./models/phi-2 \
  --judge_model ./models/Qwen2.5-7B-Instruct \
  --rebuttal_model ./models/Qwen2.5-7B-Instruct \
  --max_items 1000 \
  --mode both \
  --calib_frac 0.5 \
  --alpha 0.1 \
  --enable_rewrite \
  --backend hf \
  --out results/final_experiment_v1/suite_b_phi_2_conformal.json
