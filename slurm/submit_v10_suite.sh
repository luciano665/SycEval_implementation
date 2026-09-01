#!/bin/bash
# conformal_v10 suite (MedQuad, N=300): 6 jobs, CONFORMAL ONLY.
# Identical to v9n (deployable-only, leak-free rewrite, folding-only
# labels, global thresholds, alpha=0.10) but --threshold_method exact_crc
# instead of the Wilson-based default -- the ONE variable this suite
# isolates. No new baseline jobs: run_eval.py (baseline) never touches
# threshold fitting, so results/v9_night_medquad/run_baseline_<model>.json
# is the correct, valid baseline to pair every v10 conformal result
# against (same seed, same judge, same 300-item set).
#
# Run smoke_test_v10_gemma_1b.slurm FIRST and confirm it exits 0 with
# threshold_method/xi_certified/alpha_min populated in the thresholds file
# before submitting this.
echo "Submitting conformal_v10 suite (6 jobs, conformal only)..."
sbatch slurm/v10_llama_1b_conformal.slurm
sbatch slurm/v10_llama_3b_conformal.slurm
sbatch slurm/v10_gemma_1b_conformal.slurm
sbatch slurm/v10_gemma_4b_conformal.slurm
sbatch slurm/v10_phi_1.5_conformal.slurm
sbatch slurm/v10_phi_2_conformal.slurm
echo "All 6 jobs submitted to gpu_7day. Results -> results/v10_exact_crc_medquad/"
echo "Compare against results/v9_night_medquad/ once complete (analyze_selective_crc.py --dir results/v9_night_medquad still works for the Wilson-vs-exact retrofit; the real v10 runs give the live exact_crc numbers directly from their own thresholds_<model>.json files)."
