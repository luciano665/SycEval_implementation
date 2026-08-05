#!/bin/bash

# SycEval v6: HealthSearchQA DEPLOYABLE arm (--no_oracle_truth)
# Companion to submit_v6_suite.sh (which covers baseline + oracle-on
# conformal). Baseline has no oracle_truth concept, so it is not
# re-run here -- this only adds the oracle-off conformal comparison.

echo "Submitting SycEval v6 HealthSearchQA DEPLOYABLE arm (6 conformal jobs)..."

sbatch slurm/v6_llama_1b_conformal_no_oracle.slurm
sbatch slurm/v6_llama_3b_conformal_no_oracle.slurm
sbatch slurm/v6_gemma_1b_conformal_no_oracle.slurm
sbatch slurm/v6_gemma_4b_conformal_no_oracle.slurm
sbatch slurm/v6_phi_1.5_conformal_no_oracle.slurm
sbatch slurm/v6_phi_2_conformal_no_oracle.slurm

echo "All 6 jobs (v6-healthsearch-no_oracle) submitted to gpu_7day partition."
echo "Results will land in 'results/healthsearch_v6_1000/' with a _no_oracle suffix."
