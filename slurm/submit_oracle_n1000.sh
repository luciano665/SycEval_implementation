#!/bin/bash
# Oracle-assisted risk scorer, CALIBRATION ONLY, full suite: 6 models x 2
# datasets at the paper's N=1000 scale (200 calibration items each).
#
# Supersedes submit_oracle_diagnostic.sh, which ran the same diagnostic for 4
# models on MedQuad only at N=300. Rerunning at the primary scale, on both
# datasets, and including the two models that already certify removes the
# "N=300, MedQuad only, 4 of 6 models" caveat from the oracle comparison.
#
# Each job calibrates on the SAME 200 items the deployable suite used
# (--max_items 1000 --calib_frac 0.2), so alpha_min differences are
# attributable to the scorer seeing the reference answer and nothing else.
# Verify per model by diffing data_split.calib_question_hashes against
# results/final_n1000_<domain>/thresholds_<model>.json.
#
# Cheap relative to the final suites: mode=calibrate skips the test phase, so
# each job is roughly 1/5 of a full conformal run. 24h on gpu_2day.
#
# DIAGNOSTIC ONLY -- these thresholds have no deployable test-time path.
echo "Submitting oracle N=1000 diagnostic, MedQuad (6 jobs)..."
sbatch slurm/oracle1k_llama_1b.slurm
sbatch slurm/oracle1k_llama_3b.slurm
sbatch slurm/oracle1k_gemma_1b.slurm
sbatch slurm/oracle1k_gemma_4b.slurm
sbatch slurm/oracle1k_phi_1.5.slurm
sbatch slurm/oracle1k_phi_2.slurm

echo "Submitting oracle N=1000 diagnostic, HealthSearchQA (6 jobs)..."
sbatch slurm/oracle1khs_llama_1b.slurm
sbatch slurm/oracle1khs_llama_3b.slurm
sbatch slurm/oracle1khs_gemma_1b.slurm
sbatch slurm/oracle1khs_gemma_4b.slurm
sbatch slurm/oracle1khs_phi_1.5.slurm
sbatch slurm/oracle1khs_phi_2.slurm

echo "All 12 jobs submitted."
echo "Results -> results/oracle_n1000_medquad/ and results/oracle_n1000_healthsearch/"
echo
echo "Read each thresholds_<model>.json: alpha_min <= 0.10 means the model"
echo "certifies once the risk scorer sees the reference answer. Compare against"
echo "the deployable alpha_min in results/final_n1000_<domain>/thresholds_<model>.json:"
echo "the gap is the share of calibration failure attributable to the scorer"
echo "rather than to the tested model."
