#!/bin/bash
#SBATCH --job-name=smoke_v9_gemma_1b
#SBATCH --output=logs/output_smoke_v9_gemma_1b_%j.txt
#SBATCH --error=logs/error_smoke_v9_gemma_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# conformal_v9 smoke test (N=20, mode=both). Validates the three v9 changes
# end-to-end against a real model before committing cluster time to a full
# suite:
#   1) Deployable-only pipeline: the oracle-truth option is GONE. There is no
#      --oracle_truth / --no_oracle_truth flag anymore; the claim scorer, risk
#      scorer, and rewrite never see the reference answer.
#   2) Leak-free rewrite: the rewrite sees only the purified draft (kept
#      claims), never the rebuttal or the initial answer.
#   3) Folding-only sycophancy definition (metrics.py): correct -> erroneous
#      (refusal / parse failure) now counts as non-sycophantic, so calibration
#      should be able to certify a low-risk set instead of always rewriting.
#
# What to check after it finishes:
#   - Job exits 0, no crashes (confirms the removed flag didn't break the CLI).
#   - thresholds file: is tau_global a real number (calibration succeeded) or
#     -1.0 (still degenerate at N=20 -- expected at this tiny N, the point here
#     is that the PIPELINE runs, not that 20 items certify).
#   - conformal output: rewrite_triggered present, final_answer populated,
#     no "erroneous"-driven regressive spikes in draft_sycophancy.
set -e

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

mkdir -p results/smoke_test_v9

echo "--- conformal_v9 SMOKE (deployable-only, folding-only): N=20, mode=both ---"
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
  --domain medquad \
  --thresholds_out results/smoke_test_v9/smoke_thresholds_gemma_1b.json \
  --out results/smoke_test_v9/smoke_conformal_gemma_1b.json
echo "--- conformal phase exited 0 ---"

echo "--- SMOKE TEST DONE. Inspect: ---"
echo "  results/smoke_test_v9/smoke_conformal_gemma_1b.json"
echo "  results/smoke_test_v9/smoke_thresholds_gemma_1b.json"
echo "  results/smoke_test_v9/smoke_conformal_gemma_1b.json.calib.partial.jsonl"
echo "  results/smoke_test_v9/smoke_conformal_gemma_1b.json.test.partial.jsonl"
