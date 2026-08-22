#!/bin/bash
#SBATCH --job-name=v9n_llama_3b_base
#SBATCH --output=logs/output_v9n_llama_3b_base_%j.txt
#SBATCH --error=logs/error_v9n_llama_3b_base_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# conformal_v9 overnight suite -- BASELINE for llama_3b (MedQuad, N=300).
# Same judge (Qwen2.5-7B) and seed as the conformal arm so both share an
# identical 300-item set; pairing is on question text at analysis time.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v9_night_medquad

echo "--- v9 BASELINE llama_3b (N=300, MedQuad, Qwen judge) ---"
srun python run_eval.py \
  --tested_model models/Llama-3.2-3B-Instruct \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --backend hf \
  --max_items 300 \
  --domain medquad \
  --out results/v9_night_medquad/run_baseline_llama_3b.json
echo "--- baseline llama_3b exited 0 ---"
