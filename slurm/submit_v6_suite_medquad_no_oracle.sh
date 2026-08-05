#!/bin/bash

# SycEval v6: MedQuad DEPLOYABLE arm (--no_oracle_truth).
# Mirrors submit_v6_suite_no_oracle.sh (HealthSearchQA oracle-off).
# Companion to submit_v6_suite_medquad.sh, which covers MedQuad
# baseline + oracle-on.

echo "Submitting SycEval v6 MedQuad DEPLOYABLE arm (6 conformal jobs)..."

sbatch slurm/v6_llama_1b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_llama_3b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_gemma_1b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_gemma_4b_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_phi_1.5_conformal_medquad_no_oracle.slurm
sbatch slurm/v6_phi_2_conformal_medquad_no_oracle.slurm

echo "All 6 jobs (v6-medquad-no_oracle) submitted to gpu_7day partition."
echo "Results will land in 'results/medDataset_v6_1000/' with a _no_oracle suffix."
