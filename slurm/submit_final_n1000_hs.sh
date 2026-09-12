#!/bin/bash
# Final experiment on the second dataset: HealthSearchQA, N=1000
# (200 calib / 800 test), all 6 models, baseline + conformal = 12 jobs.
#
# Identical protocol to the MedQuad final suite (submit_final_n1000.sh) --
# deployable-only, leak-free rewrite, folding-only labels, alpha=0.10,
# threshold_method=exact_crc, calib_frac=0.2 -- with only --domain
# changed. Holding every other knob fixed is deliberate: it makes the two
# datasets directly comparable.
#
# The old v6-era HealthSearchQA results are NOT a substitute for this run
# (review_findings.md C1: oracle truth leaked into the test-time
# intervention in that pipeline, which is what inflated the retracted
# "7.4% reduction" figure).
#
# Requires data/healthsearch_qa.jsonl to be present (3047 rows); the
# loader raises FileNotFoundError if it is missing.
set -e

if [ ! -f data/healthsearch_qa.jsonl ]; then
  echo "ERROR: data/healthsearch_qa.jsonl not found. Aborting before submitting anything." >&2
  exit 1
fi

echo "Submitting final HealthSearchQA N=1000 suite (12 jobs)..."
sbatch slurm/final1khs_llama_1b_baseline.slurm
sbatch slurm/final1khs_llama_1b_conformal.slurm
sbatch slurm/final1khs_llama_3b_baseline.slurm
sbatch slurm/final1khs_llama_3b_conformal.slurm
sbatch slurm/final1khs_gemma_1b_baseline.slurm
sbatch slurm/final1khs_gemma_1b_conformal.slurm
sbatch slurm/final1khs_gemma_4b_baseline.slurm
sbatch slurm/final1khs_gemma_4b_conformal.slurm
sbatch slurm/final1khs_phi_1.5_baseline.slurm
sbatch slurm/final1khs_phi_1.5_conformal.slurm
sbatch slurm/final1khs_phi_2_baseline.slurm
sbatch slurm/final1khs_phi_2_conformal.slurm
echo "All 12 jobs submitted. Results -> results/final_n1000_healthsearch/"
