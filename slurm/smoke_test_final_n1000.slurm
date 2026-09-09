#!/bin/bash
#SBATCH --job-name=smoke_final1k
#SBATCH --output=logs/output_smoke_final1k_%j.txt
#SBATCH --error=logs/error_smoke_final1k_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day

# Smoke test for the final N=1000 suite's exact flag combination before
# committing multi-day compute to the real run. calib_frac=0.2 has never
# been used before (every prior run used 0.5) -- confirms the split
# arithmetic and full pipeline work correctly with this new value,
# combined with threshold_method=exact_crc, at a tiny N=20
# (calib_frac=0.2 -> 4 calib / 16 test) before scaling to N=1000.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/smoke_test_final1k

echo "--- final N=1000 suite smoke test: gemma_1b (N=20, calib_frac=0.2, exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.2 \
  --max_items 20 \
  --enable_rewrite \
  --domain medquad \
  --threshold_method exact_crc \
  --thresholds_out results/smoke_test_final1k/smoke_thresholds_gemma_1b.json \
  --out results/smoke_test_final1k/smoke_conformal_gemma_1b.json
echo "--- exited 0 ---"
echo "--- check: data_split should show ~4 calib / ~16 test items (calib_frac=0.2 of N=20) ---"
