#!/bin/bash
#SBATCH --job-name=v9j_phi_2_base
#SBATCH --output=logs/output_v9j_phi_2_base_%j.txt
#SBATCH --error=logs/error_v9j_phi_2_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# Independent-judge suite -- BASELINE for phi_2 (MedQuad, N=300).
# Same structure as v9n_phi_2_baseline.slurm, but judge_model swapped to
# Mistral-7B-Instruct-v0.3 (independent of rebuttal_model=Qwen2.5-7B) to
# check for judge self-preference bias. Same seed/N as the conformal arm
# so both share an identical 300-item set; pairing is on question text at
# analysis time.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_judge_split_medquad

echo "--- referee-split BASELINE phi_2 (N=300, MedQuad, Mistral judge) ---"
srun python run_eval.py \
  --tested_model models/phi-2 \
  --judge_model models/Mistral-7B-Instruct-v0.3 \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v9_judge_split_medquad/run_baseline_phi_2.json
echo "--- baseline phi_2 exited 0 ---"
