#!/bin/bash
#SBATCH --job-name=neutral_qwen_1.5B_conf
#SBATCH --output=logs/neutral_qwen_1.5B_conf_%j.txt
#SBATCH --error=logs/error_neutral_qwen_1.5B_conf_%j.txt
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

# Run Qwen 1.5B Conformal with Ministral Judge/Rebuttal
srun python -m conformal_v2.run_conformal_v2 \
  --backend hf \
  --tested_model ./models/Qwen2.5-1.5B-Instruct \
  --judge_model ./models/Ministral-8B-Instruct-2410 \
  --rebuttal_model ./models/Ministral-8B-Instruct-2410 \
  --alpha 0.1 \
  --calib_frac 0.2 \
  --max_items 1000 \
  --out results/neutral_suite_v1/qwen_1.5B_conformal.json
