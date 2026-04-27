#!/bin/bash

# SycEval v5: COMPLETE SUITE (Baseline + Conformal Skeptic)
# Methodology: N=300, MedQuad Only, Delta-Sycophancy Scorer, Alpha=0.05, Group-Thresholds.

echo "🚀 Submitting SycEval v5 FULL SUITE (12 jobs)..."

# --- BASELINES ---
sbatch slurm/v5_llama_1b_baseline.slurm
sbatch slurm/v5_llama_3b_baseline.slurm
sbatch slurm/v5_gemma_1b_baseline.slurm
sbatch slurm/v5_gemma_4b_baseline.slurm
sbatch slurm/v5_phi_1.5_baseline.slurm
sbatch slurm/v5_phi_2_baseline.slurm

# --- CONFORMAL SKEPTIC ---
sbatch slurm/v5_llama_1b_conformal.slurm
sbatch slurm/v5_llama_3b_conformal.slurm
sbatch slurm/v5_gemma_1b_conformal.slurm
sbatch slurm/v5_gemma_4b_conformal.slurm
sbatch slurm/v5_phi_1.5_conformal.slurm
sbatch slurm/v5_phi_2_conformal.slurm

echo "✅ All 12 jobs (v5) submitted to gpu_7day partition."
echo "Results will aggregate in 'results/final_experiment_v5/'"
