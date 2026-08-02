#!/bin/bash
#SBATCH --job-name=smoke_gemma_1b_hs
#SBATCH --output=logs/output_smoke_gemma_1b_hs_%j.txt
#SBATCH --error=logs/error_smoke_gemma_1b_hs_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# Second-stage smoke test: closes two gaps the first round (medquad, no
# group thresholds) didn't cover. The REAL submit_v6_suite.sh runs
# exclusively on --domain healthsearch and always passes
# --use_group_thresholds -- neither of those code paths has been exercised
# against real models yet, even though the earlier medquad smoke tests all
# passed. N=20 is still too small for group thresholds to be meaningful
# (C4's own docstring: ~70+ clean items per group needed for 24 groups) --
# this run is about confirming the CODE PATH runs without error, not about
# getting a real threshold. Expect the same kind of degenerate-calibration
# ERROR logging we saw before (C2), now exercised on healthsearch domain.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test

echo "--- SMOKE TEST (healthsearch): baseline (N=20) ---"
srun python run_eval.py \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 20 \
  --domain healthsearch \
  --out results/smoke_test/smoke_baseline_gemma_1b_hs.json
echo "--- baseline phase exited 0 ---"

echo "--- SMOKE TEST (healthsearch): conformal (N=20, mode=both, use_group_thresholds) ---"
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
  --use_group_thresholds \
  --oracle_truth \
  --domain healthsearch \
  --thresholds_out results/smoke_test/smoke_thresholds_gemma_1b_hs.json \
  --out results/smoke_test/smoke_conformal_gemma_1b_hs.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE (both phases exited 0). Inspect: ---"
echo "  results/smoke_test/smoke_baseline_gemma_1b_hs.json"
echo "  results/smoke_test/smoke_conformal_gemma_1b_hs.json"
echo "  results/smoke_test/smoke_thresholds_gemma_1b_hs.json"
echo "  results/smoke_test/smoke_conformal_gemma_1b_hs.json.calib.partial.jsonl"
echo "  results/smoke_test/smoke_conformal_gemma_1b_hs.json.test.partial.jsonl"
