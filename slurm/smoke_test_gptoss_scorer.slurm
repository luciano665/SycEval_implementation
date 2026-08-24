#!/bin/bash
#SBATCH --job-name=smoke_gptoss_scorer
#SBATCH --output=logs/output_smoke_gptoss_scorer_%j.txt
#SBATCH --error=logs/error_smoke_gptoss_scorer_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_2day

# Smoke test: gpt-oss-20b as risk_scorer_model (judge/rebuttal stay Qwen --
# isolating the scorer-model-capability question from everything else).
# N=20 only -- this run exists to catch two specific unknowns before
# spending an hour on the real N=60 comparison:
#   1) VRAM: gpt-oss-20b ships natively in MXFP4 (~16GB per OpenAI's card),
#      but models.py forces dtype=bf16/fp16 on load -- if it can't use the
#      native quantized weights and upcasts, real usage could be ~42GB.
#      Pinned to an A100 node here specifically because we don't yet know
#      which of these actually happens.
#   2) Chat format: gpt-oss uses OpenAI's "harmony" response format, not
#      plain system/user/assistant. parse_float_0_1() looks for a literal
#      "Final Score:" line -- if harmony wraps the output differently,
#      every score could silently default to 1.0 (max risk) via the
#      parse-failure fallback, not a crash, so must inspect real output.
#
# What to check after it finishes:
#   - Job exits 0 (confirms the model loads and runs at all on this node).
#   - Actual risk_score values in the output are NOT all exactly 1.0
#     (that pattern specifically would mean parse_float_0_1 is silently
#     failing to find "Final Score:" in gpt-oss's real output format).
#   - Spot-check a few raw scorer outputs if possible to see the real
#     response shape.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/smoke_test_gptoss_scorer

echo "--- gpt-oss-20b scorer SMOKE (judge/rebuttal=Qwen): N=20, mode=both ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/gpt-oss-20b \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 20 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/smoke_test_gptoss_scorer/smoke_thresholds_llama_1b.json \
  --out results/smoke_test_gptoss_scorer/smoke_conformal_llama_1b.json
echo "--- exited 0 ---"

echo "--- SMOKE TEST DONE. Inspect: ---"
echo "  results/smoke_test_gptoss_scorer/smoke_conformal_llama_1b.json.calib.partial.jsonl"
