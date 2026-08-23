#!/bin/bash
#SBATCH --job-name=smoke_referee_split_gemma_1b
#SBATCH --output=logs/output_smoke_referee_split_gemma_1b_%j.txt
#SBATCH --error=logs/error_smoke_referee_split_gemma_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# Referee-split smoke test (N=20, mode=both). Validates the independent-judge
# change end-to-end against real models before committing cluster time to a
# full suite:
#   - judge_model    = Mistral-7B-Instruct-v0.3  (NEW: independent of rebuttal/scorer)
#   - rebuttal_model  = Qwen2.5-7B-Instruct       (unchanged from v9)
#   - risk_scorer_model = Qwen2.5-7B-Instruct     (explicit -- must NOT be left to
#     default, or it silently inherits judge_model=Mistral via the
#     `args.risk_scorer_model.strip() or cfg.judge_model` fallback in
#     run_conformal_v2.py)
#
# What to check after it finishes:
#   - Job exits 0 (confirms Mistral loads correctly through models.py's
#     from_pretrained path -- new model family, not smoke-tested before).
#   - thresholds file / conformal output: judge-driven labels look sane
#     (spot check a few records), risk scores populated and NOT identical
#     to what a Qwen-judge run would produce (sanity that Mistral is
#     actually being called for judging, not silently falling back).
#   - No unexpected errors in the Mistral-specific config-fallback path in
#     models.py (it has custom handling for Mistral/Pixtral config quirks --
#     confirm it wasn't needed, or if it was, confirm it worked).
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test_referee_split

echo "--- referee-split SMOKE (judge=Mistral, rebuttal/scorer=Qwen): N=20, mode=both ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/gemma-3-1b-it \
  --judge_model models/Mistral-7B-Instruct-v0.3 \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 20 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/smoke_test_referee_split/smoke_thresholds_gemma_1b.json \
  --out results/smoke_test_referee_split/smoke_conformal_gemma_1b.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE. Inspect: ---"
echo "  results/smoke_test_referee_split/smoke_conformal_gemma_1b.json"
echo "  results/smoke_test_referee_split/smoke_thresholds_gemma_1b.json"
echo "  results/smoke_test_referee_split/smoke_conformal_gemma_1b.json.calib.partial.jsonl"
echo "  results/smoke_test_referee_split/smoke_conformal_gemma_1b.json.test.partial.jsonl"
