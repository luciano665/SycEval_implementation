#!/bin/bash
#SBATCH --job-name=smoke_gemma_1b_no_oracle
#SBATCH --output=logs/output_smoke_gemma_1b_no_oracle_%j.txt
#SBATCH --error=logs/error_smoke_gemma_1b_no_oracle_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# Third-stage smoke test: --no_oracle_truth has never been exercised against
# real models -- every prior smoke test used --oracle_truth. This routes
# through genuinely different prompts in scorers.py / syco_risk.py /
# safe_rewrite.py (none of them see the reference answer). Domain is medquad
# (already validated separately, job 123393) so oracle_truth is the ONLY
# new variable being tested here -- baseline is skipped entirely since it
# has no oracle_truth concept at all (only conformal_v2 does).
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test

echo "--- SMOKE TEST (no_oracle_truth, medquad): conformal (N=20, mode=both) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.5 \
  --max_items 20 \
  --enable_rewrite \
  --no_oracle_truth \
  --domain medquad \
  --thresholds_out results/smoke_test/smoke_thresholds_gemma_1b_no_oracle.json \
  --out results/smoke_test/smoke_conformal_gemma_1b_no_oracle.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE. Inspect: ---"
echo "  results/smoke_test/smoke_conformal_gemma_1b_no_oracle.json"
echo "  results/smoke_test/smoke_thresholds_gemma_1b_no_oracle.json"
echo "  results/smoke_test/smoke_conformal_gemma_1b_no_oracle.json.calib.partial.jsonl"
echo "  results/smoke_test/smoke_conformal_gemma_1b_no_oracle.json.test.partial.jsonl"
