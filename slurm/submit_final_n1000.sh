#!/bin/bash
# Final experiment for the paper: N=1000 (200 calib / 800 test), all 6
# models, baseline + conformal = 12 jobs. Same protocol as v9n/v10
# (deployable-only, leak-free rewrite, folding-only labels, alpha=0.10),
# using threshold_method=exact_crc (the more rigorous bound). Split
# chosen via analyze_calib_size_projection.py against real v9 data: 200
# calibration items already captures the full achievable benefit for the
# one borderline model it affects (Gemma-1B); the rest goes to test-set
# precision, which keeps improving with more data unlike calibration.
echo "Submitting final N=1000 experiment suite (12 jobs)..."
sbatch slurm/final1k_llama_1b_baseline.slurm
sbatch slurm/final1k_llama_1b_conformal.slurm
sbatch slurm/final1k_llama_3b_baseline.slurm
sbatch slurm/final1k_llama_3b_conformal.slurm
sbatch slurm/final1k_gemma_1b_baseline.slurm
sbatch slurm/final1k_gemma_1b_conformal.slurm
sbatch slurm/final1k_gemma_4b_baseline.slurm
sbatch slurm/final1k_gemma_4b_conformal.slurm
sbatch slurm/final1k_phi_1.5_baseline.slurm
sbatch slurm/final1k_phi_1.5_conformal.slurm
sbatch slurm/final1k_phi_2_baseline.slurm
sbatch slurm/final1k_phi_2_conformal.slurm
echo "All 12 jobs submitted. Results -> results/final_n1000_medquad/"
