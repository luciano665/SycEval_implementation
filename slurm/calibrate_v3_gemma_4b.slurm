#!/bin/bash
#SBATCH --job-name=calib_v3_gemma_4b
#SBATCH --output=logs/output_calib_v3_gemma_4b_%j.txt
#SBATCH --error=logs/error_calib_v3_gemma_4b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_7day

# "Quicker test first" checkpoint (moderate N=60 items / ~480 records, not
# the full 250-item real-suite scale): does v3 scoring actually let
# fit_global_threshold succeed for Gemma-4B, or does the AUC improvement
# (0.60 -> 0.85 on the small N=120 diagnostic) not translate into real
# calibration success? Directional signal -- if promising, the natural
# follow-up is the same test at full 250-item scale.

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test

srun python calibrate_with_v3_scorer.py \
  --tested_model models/gemma-3-4b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain healthsearch \
  --oracle_truth \
  --n_items 60 \
  --alpha 0.05 \
  --checkpoint_path results/smoke_test/calib_v3_gemma_4b.jsonl \
  > results/smoke_test/calibrate_v3_gemma_4b_output.txt

echo "--- DONE. Output saved to results/smoke_test/calibrate_v3_gemma_4b_output.txt ---"
