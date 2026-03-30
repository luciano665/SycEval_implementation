#!/bin/bash

# SycEval v6: COMPLETE ADVERSARIAL SUITE (HealthSearchQA Expansion)
# Methodology: N=1000 (calib=250, test=750), HealthSearchQA Domain, Chain-of-Thought Scorer, Alpha=0.05, Group-Thresholds.

echo "🚀 Submitting SycEval v6 FULL ADVERSARIAL SUITE (12 jobs)..."

# --- BASELINES ---
sbatch slurm/v6_llama_1b_baseline.slurm
sbatch slurm/v6_llama_3b_baseline.slurm
sbatch slurm/v6_gemma_1b_baseline.slurm
sbatch slurm/v6_gemma_4b_baseline.slurm
sbatch slurm/v6_phi_1.5_baseline.slurm
sbatch slurm/v6_phi_2_baseline.slurm

# --- CONFORMAL ADVERSARIAL OVERSIGHT ---
sbatch slurm/v6_llama_1b_conformal.slurm
sbatch slurm/v6_llama_3b_conformal.slurm
sbatch slurm/v6_gemma_1b_conformal.slurm
sbatch slurm/v6_gemma_4b_conformal.slurm
sbatch slurm/v6_phi_1.5_conformal.slurm
sbatch slurm/v6_phi_2_conformal.slurm

echo "✅ All 12 jobs (v6-healthsearch) submitted to gpu_7day partition."
echo "Results will aggregate in 'results/healthsearch_v6_1000/'"
