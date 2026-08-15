#!/bin/bash
#SBATCH --job-name=diag_risk_v2_llama_1b
#SBATCH --output=logs/output_diag_risk_v2_llama_1b_%j.txt
#SBATCH --error=logs/error_diag_risk_v2_llama_1b_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --partition=gpu_7day

# Controlled comparison, not an experiment: same fresh items scored under
# BOTH the current risk-scorer prompt and the revised one (calibration
# anchors + explicit maintained/declined/adopted bands), to check whether
# the revision actually improves separation (mean-score gap, AUC) and
# reduces the "refusals scored as high risk" ceiling-effect symptom --
# before deciding whether to adopt it in the real pipeline.

source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME

srun python diagnose_risk_scorer_v2.py \
  --tested_model models/Llama-3.2-1B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --domain healthsearch \
  --oracle_truth \
  --n_items 15 \
  > results/smoke_test/diagnose_risk_scorer_v2_llama_1b_output.txt

echo "--- DONE. Output saved to results/smoke_test/diagnose_risk_scorer_v2_llama_1b_output.txt ---"
