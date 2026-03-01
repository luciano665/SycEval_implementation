#!/bin/bash

# SycEval v3: Unified Submission Script
# This script submits all 12 experiments (6 Baselines + 6 Conformal) for the Truth-Grounded Evaluation.

echo "🚀 Submitting SycEval v3 Truth-Grounded Suite..."

# 1. Llama Family
sbatch slurm/v3_llama_1b_baseline.slurm
sbatch slurm/v3_llama_1b_conformal.slurm
sbatch slurm/v3_llama_3b_baseline.slurm
sbatch slurm/v3_llama_3b_conformal.slurm

# 2. Gemma Family
sbatch slurm/v3_gemma_1b_baseline.slurm
sbatch slurm/v3_gemma_1b_conformal.slurm
sbatch slurm/v3_gemma_4b_baseline.slurm
sbatch slurm/v3_gemma_4b_conformal.slurm

# 3. Phi Family
sbatch slurm/v3_phi_1.5_baseline.slurm
sbatch slurm/v3_phi_1.5_conformal.slurm
sbatch slurm/v3_phi_2_baseline.slurm
sbatch slurm/v3_phi_2_conformal.slurm

echo "✅ All jobs submitted. Check progress with 'squeue -u \$USER' or monitor output in 'logs/'."
