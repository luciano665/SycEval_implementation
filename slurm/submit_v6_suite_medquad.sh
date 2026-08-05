#!/bin/bash

# SycEval v6: COMPLETE MedQuad SUITE (mirrors submit_v6_suite.sh, which
# covers HealthSearchQA). N=1000 (calib=250, test=750), Alpha=0.05,
# Group-Thresholds. Runs BOTH oracle_truth arms so the oracle gap can be
# measured on MedQuad the same way as HealthSearchQA.

echo "Submitting SycEval v6 MedQuad FULL SUITE (18 jobs)..."

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

# --- CONFORMAL, DEPLOYABLE (--no_oracle_truth) ---
sbatch slurm/v6_llama_1b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_llama_3b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_gemma_1b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_gemma_4b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_phi_1.5_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_phi_2_conformal_medquad_no_oracle.slurm

echo "All 18 jobs (v6-medquad) submitted to gpu_7day partition."
echo "Results will aggregate in 'results/medDataset_v6_1000/'"
