#!/bin/bash
#SBATCH --job-name=v10_phi_2_conf
#SBATCH --output=logs/output_v10_phi_2_conf_%j.txt
#SBATCH --error=logs/error_v10_phi_2_conf_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu_7day

# conformal_v10 -- CONFORMAL (CCR) for phi_2 (MedQuad, N=300).
# Identical to v9n except ONE variable: --threshold_method exact_crc
# instead of the Wilson-based default. Phi-2 already certifies under
# Wilson (tau_global=1.0 in v9n) -- included mainly as a control: exact_crc
# should NOT change an already-successful, non-borderline certification.
# Same seed/judge/rebuttal models/data split as
# results/v9_night_medquad/run_conformal_phi_2.json.
#
# No new baseline needed: results/v9_night_medquad/run_baseline_phi_2.json
# (already on disk) is the correct, valid baseline to pair this against.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$SCRATCH/hf_home
mkdir -p $HF_HOME
mkdir -p results/v10_exact_crc_medquad

echo "--- v10 CONFORMAL phi_2 (N=300, alpha=0.10, threshold_method=exact_crc) ---"
srun python -m conformal_v2.run_conformal_v2 \
  --mode both \
  --backend hf \
  --tested_model models/phi-2 \
  --judge_model models/Qwen2.5-7B-Instruct \
  --rebuttal_model models/Qwen2.5-7B-Instruct \
  --alpha 0.10 \
  --calib_frac 0.5 \
  --max_items 300 \
  --enable_rewrite \
  --domain medquad \
  --threshold_method exact_crc \
  --thresholds_out results/v10_exact_crc_medquad/thresholds_phi_2.json \
  --out results/v10_exact_crc_medquad/run_conformal_phi_2.json
echo "--- conformal phi_2 exited 0 ---"
