#!/bin/bash
#SBATCH --job-name=verify_gptoss_llama_1b
#SBATCH --output=logs/output_verify_gptoss_llama_1b_%j.txt
#SBATCH --error=logs/error_verify_gptoss_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:nvidia_a100:2
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_2day

# Verifies whether gpt-oss-20b as risk_scorer_model closes the AUC gap
# that v3's prompt fix couldn't (llama_1b: AUC 0.470, still calibration_failed).
# judge_model/rebuttal_model stay Qwen (unchanged from every prior run) so
# only the risk-scorer-model variable changes, isolating model capability
# from everything else already tested.
#
# N=60 (30 calib items -> 240 calibration instances), same scale as the
# v3 verification runs for direct comparison. Smoke-tested first at N=20
# (job 136414, COMPLETED, real varied scores confirmed -- not stuck on
# harmony-format parsing) before committing to this run.
#
# 2 GPUs + the multi-GPU sharding patch in models.py::_ensure_hf: this
# cluster's env lacks Triton for gpt-oss's native MXFP4, so it
# dequantizes to bf16 (~42GB) -- doesn't fit on 1 GPU (confirmed OOM on
# both A30 24GB and this cluster's A100). HF_HOME points at the project
# filesystem, not $SCRATCH (not mounted on A100 nodes).
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME
mkdir -p results/verify_gptoss_scorer

echo "--- gpt-oss-20b scorer verification: llama_1b (N=60) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --risk_scorer_model models/gpt-oss-20b \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 60 \
  --enable_rewrite \
  --domain medquad \
  --thresholds_out results/verify_gptoss_scorer/thresholds_llama_1b.json \
  --out results/verify_gptoss_scorer/run_conformal_llama_1b.json
echo "--- llama_1b exited 0 ---"
