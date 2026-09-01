#!/bin/bash
# Mentor-requested diagnostic (see docs/FINAL_DIAGNOSTICS_TRACKER.md):
# oracle-assisted risk scorer, CALIBRATION ONLY, for the 4 models whose
# calibration currently fails (Llama-1B/3B, Gemma-1B/4B). Phi-1.5/2 already
# certify -- not included, nothing to diagnose there.
#
# Cheap relative to the v9/v10 suites: mode=calibrate skips the test phase
# entirely, so this is 4 jobs, each much shorter than a full "both" run.
echo "Submitting oracle-assisted risk scorer diagnostic (4 jobs, calibration only)..."
sbatch slurm/oracle_risk_scorer_calib_llama_1b.slurm
sbatch slurm/oracle_risk_scorer_calib_llama_3b.slurm
sbatch slurm/oracle_risk_scorer_calib_gemma_1b.slurm
sbatch slurm/oracle_risk_scorer_calib_gemma_4b.slurm
echo "All 4 jobs submitted. Results -> results/oracle_diagnostic_medquad/"
echo "Check each thresholds_<model>.json: tau_global still -1.0 (fails even with"
echo "the oracle) vs a real number (oracle rescues it -- scorer WAS the bottleneck)."
