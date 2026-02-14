#!/bin/bash
# Targeted Submission Script for Suite B Gaps
# Methodology: Qwen-7B Judge, 0.5 Calib Fraction, Mixed Data

submit() {
    echo "Submitting $1..."
    sbatch "slurm/$1"
}

echo "=== Launching Targeted Suite B Runs ==="

# Phi Family (Baseline and Conformal)
submit "suite_b_phi_1.5_baseline.slurm"
submit "suite_b_phi_1.5_conformal.slurm"
submit "suite_b_phi_2_baseline.slurm"
submit "suite_b_phi_2_conformal.slurm"

# 1B Baselines
submit "suite_b_llama_1b_baseline.slurm"
submit "suite_b_gemma_1b_baseline.slurm"

echo "✅ Targeted Suite B jobs submitted!"
