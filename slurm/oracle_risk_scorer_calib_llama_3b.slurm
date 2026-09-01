#!/bin/bash
#SBATCH --job-name=oracle_calib_llama_3b
#SBATCH --output=logs/output_oracle_calib_llama_3b_%j.txt
#SBATCH --error=logs/error_oracle_calib_llama_3b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_2day

# Mentor-requested diagnostic -- see oracle_risk_scorer_calib_llama_1b.slurm
# for the full explanation. --max_items 150 matches v9n/v10's calibration
# slice size exactly.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/oracle_diagnostic_medquad

echo "--- oracle-assisted risk scorer, CALIBRATION ONLY, llama_3b (N=150, alpha=0.10) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode calibrate \
  --backend hf \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --max_items 150 \
  --domain medquad \
  --threshold_method exact_crc \
  --oracle_risk_scorer_calib \
  --thresholds_out results/oracle_diagnostic_medquad/thresholds_llama_3b.json \
  --out results/oracle_diagnostic_medquad/calib_llama_3b.json
echo "--- oracle-calibration llama_3b exited 0 ---"
