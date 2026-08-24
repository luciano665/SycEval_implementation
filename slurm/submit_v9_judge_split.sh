#!/bin/bash
# Independent-judge suite (MedQuad, N=300): 12 jobs (6 baseline + 6 conformal).
# Same as v9n (deployable-only, leak-free rewrite, folding-only labels,
# global thresholds, alpha=0.10) but judge_model=Mistral-7B-Instruct-v0.3,
# independent of rebuttal_model/risk_scorer_model=Qwen2.5-7B-Instruct.
# Verified via smoke test (job 135218, N=20, exit 0:0) before this launch.
echo "Submitting independent-judge suite (12 jobs)..."
sbatch slurm/v9j_llama_1b_baseline.slurm
sbatch slurm/v9j_llama_1b_conformal.slurm
sbatch slurm/v9j_llama_3b_baseline.slurm
sbatch slurm/v9j_llama_3b_conformal.slurm
sbatch slurm/v9j_gemma_1b_baseline.slurm
sbatch slurm/v9j_gemma_1b_conformal.slurm
sbatch slurm/v9j_gemma_4b_baseline.slurm
sbatch slurm/v9j_gemma_4b_conformal.slurm
sbatch slurm/v9j_phi_1.5_baseline.slurm
sbatch slurm/v9j_phi_1.5_conformal.slurm
sbatch slurm/v9j_phi_2_baseline.slurm
sbatch slurm/v9j_phi_2_conformal.slurm
echo "All 12 jobs submitted to gpu_7day. Results -> results/v9_judge_split_medquad/"
