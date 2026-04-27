#!/bin/bash

# SycEval v4: Standard Suite Submission Script
# This script submits all 6 experiments (N=300, MedQuad Only).

echo "🚀 Submitting SycEval v4 Suite (N=300, MedQuad Only)..."

# 1. Llama Family
sbatch slurm/v4_llama_1b_baseline.slurm
sbatch slurm/v4_llama_1b_conformal.slurm
sbatch slurm/v4_llama_3b_baseline.slurm
sbatch slurm/v4_llama_3b_conformal.slurm

# 2. Gemma Family
sbatch slurm/v4_gemma_1b_baseline.slurm
sbatch slurm/v4_gemma_1b_conformal.slurm
sbatch slurm/v4_gemma_4b_baseline.slurm
sbatch slurm/v4_gemma_4b_conformal.slurm

# 3. Phi Family
sbatch slurm/v4_phi_1.5_baseline.slurm
sbatch slurm/v4_phi_1.5_conformal.slurm
sbatch slurm/v4_phi_2_baseline.slurm
sbatch slurm/v4_phi_2_conformal.slurm

echo "✅ All jobs submitted. Check progress with 'squeue -u \$USER' or monitor output in 'logs/'."
