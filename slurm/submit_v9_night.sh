#!/bin/bash
# conformal_v9 overnight suite (MedQuad, N=300): 12 jobs (6 baseline + 6 conformal).
# Deployable-only + leak-free rewrite + folding-only labels + global thresholds, alpha=0.10.
echo "Submitting conformal_v9 overnight suite (12 jobs)..."
sbatch slurm/v9n_llama_1b_baseline.slurm
sbatch slurm/v9n_llama_1b_conformal.slurm
sbatch slurm/v9n_llama_3b_baseline.slurm
sbatch slurm/v9n_llama_3b_conformal.slurm
sbatch slurm/v9n_gemma_1b_baseline.slurm
sbatch slurm/v9n_gemma_1b_conformal.slurm
sbatch slurm/v9n_gemma_4b_baseline.slurm
sbatch slurm/v9n_gemma_4b_conformal.slurm
sbatch slurm/v9n_phi_1.5_baseline.slurm
sbatch slurm/v9n_phi_1.5_conformal.slurm
sbatch slurm/v9n_phi_2_baseline.slurm
sbatch slurm/v9n_phi_2_conformal.slurm
echo "All 12 jobs submitted to gpu_7day. Results -> results/v9_night_medquad/"
