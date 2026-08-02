#!/bin/bash
#SBATCH --job-name=smoke_llama_3b
#SBATCH --output=logs/output_smoke_llama_3b_%j.txt
#SBATCH --error=logs/error_smoke_llama_3b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# Smoke test for the ccr-fixes / cleanup-unused-files branch: small-N sanity
# check that the fixed pipeline (C1-C13) actually runs end to end against
# the real models/backend before committing 7-day cluster time to the full
# V6 suite. Not meant to produce reportable numbers.
#
# set -e: without this, a failed srun (e.g. the compute node can't reach
# huggingface.co and the model path doesn't resolve locally) does not stop
# the script -- bash just continues to the next command and the final "DONE"
# echo prints regardless, masking the failure.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test

echo "--- SMOKE TEST: baseline (N=20, medquad) ---"
srun python run_eval.py \
  --backend hf \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --max_items 20 \
  --domain medquad \
  --out results/smoke_test/smoke_baseline_llama_3b.json
echo "--- baseline phase exited 0 ---"

echo "--- SMOKE TEST: conformal (N=20, medquad, mode=both, no group thresholds) ---"
# --use_group_thresholds is omitted: 20 items can't populate 24 groups
# meaningfully (C4's own docstring warns ~70+ clean items per group needed).
# --oracle_truth is the historical V6 default; swap to --no_oracle_truth to
# smoke-test the deployable variant instead.
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.05 \
  --calib_frac 0.5 \
  --max_items 20 \
  --enable_rewrite \
  --oracle_truth \
  --domain medquad \
  --thresholds_out results/smoke_test/smoke_thresholds_llama_3b.json \
  --out results/smoke_test/smoke_conformal_llama_3b.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE (both phases exited 0). Inspect: ---"
echo "  results/smoke_test/smoke_baseline_llama_3b.json"
echo "  results/smoke_test/smoke_conformal_llama_3b.json"
echo "  results/smoke_test/smoke_thresholds_llama_3b.json"
echo "  results/smoke_test/smoke_conformal_llama_3b.json.calib.partial.jsonl"
echo "  results/smoke_test/smoke_conformal_llama_3b.json.test.partial.jsonl"
