#!/bin/bash

# SycEval v6: MedQuad SUITE -- baseline + oracle-assisted upper bound.
# Mirrors submit_v6_suite.sh (HealthSearchQA baseline + oracle-on).
# The deployable (--no_oracle_truth) arm is a separate script:
# submit_v6_suite_medquad_no_oracle.sh -- same split as HealthSearchQA's
# submit_v6_suite.sh / submit_v6_suite_no_oracle.sh pair.

echo "Submitting SycEval v6 MedQuad suite: baseline + oracle-on (12 jobs)..."

# --- BASELINES ---
sbatch slurm/v6_llama_1b_baseline_medquad.slurm
sbatch slurm/v6_llama_3b_baseline_medquad.slurm
sbatch slurm/v6_gemma_1b_baseline_medquad.slurm
sbatch slurm/v6_gemma_4b_baseline_medquad.slurm
sbatch slurm/v6_phi_1.5_baseline_medquad.slurm
sbatch slurm/v6_phi_2_baseline_medquad.slurm

# --- CONFORMAL, ORACLE-ASSISTED UPPER BOUND (--oracle_truth) ---
sbatch slurm/v6_llama_1b_conformal_medquad_oracle.slurm
sbatch slurm/v6_llama_3b_conformal_medquad_oracle.slurm
sbatch slurm/v6_gemma_1b_conformal_medquad_oracle.slurm
sbatch slurm/v6_gemma_4b_conformal_medquad_oracle.slurm
sbatch slurm/v6_phi_1.5_conformal_medquad_oracle.slurm
sbatch slurm/v6_phi_2_conformal_medquad_oracle.slurm

echo "All 12 jobs (v6-medquad baseline + oracle-on) submitted to gpu_7day partition."
echo "Results will aggregate in 'results/medDataset_v6_1000/'"
