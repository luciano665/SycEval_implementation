#!/bin/bash
# conformal_v10 suite (MedQuad, N=300): 6 CONFORMAL jobs by default.
# Identical to v9n (deployable-only, leak-free rewrite, folding-only
# labels, global thresholds, alpha=0.10) but --threshold_method exact_crc
# instead of the Wilson-based default -- the ONE variable this suite
# isolates.
#
# No new baseline jobs BY DEFAULT: run_eval.py (baseline) never touches
# threshold fitting, so results/v9_night_medquad/run_baseline_<model>.json
# is the correct, valid baseline to pair every v10 conformal result
# against (same seed, same judge, same 300-item set) -- reusing it saves
# 6 multi-hour GPU jobs that would just reproduce numbers already on disk.
#
# Preflight check below verifies those 6 v9 baseline files actually exist
# before submitting anything. If any are missing (deleted, never synced,
# whatever), this ABORTS instead of silently leaving you with 6
# conformal-only results that have nothing to pair against -- submit the
# matching slurm/v10_<model>_baseline.slurm fallback scripts instead (or
# all of them: `for f in slurm/v10_*_baseline.slurm; do sbatch "$f"; done`).
#
# Run smoke_test_v10_gemma_1b.slurm FIRST and confirm it exits 0 with
# threshold_method/xi_certified/alpha_min populated in the thresholds file
# before submitting this.
set -e

MODELS="llama_1b llama_3b gemma_1b gemma_4b phi_1.5 phi_2"
V9_BASE_DIR="results/v9_night_medquad"
missing=0
for m in $MODELS; do
    f="$V9_BASE_DIR/run_baseline_${m}.json"
    if [ ! -f "$f" ]; then
        echo "MISSING: $f"
        missing=1
    fi
done

if [ "$missing" -eq 1 ]; then
    echo ""
    echo "ABORTING: not all v9 baseline files are present -- v10's conformal-only"
    echo "plan depends on pairing against them. Either restore/sync $V9_BASE_DIR,"
    echo "or submit the fallback baseline jobs first/instead:"
    echo "  for f in slurm/v10_*_baseline.slurm; do sbatch \"\$f\"; done"
    echo "then re-run this script once those complete (it will pass the check"
    echo "against results/v9_night_medquad if restored, or you can point"
    echo "analysis scripts at results/v10_exact_crc_medquad's own baseline"
    echo "files instead if you ran the fallback)."
    exit 1
fi

echo "All 6 v9 baseline files found -- safe to reuse them, no baseline rerun needed."
echo "Submitting conformal_v10 suite (6 jobs, conformal only)..."
sbatch slurm/v10_llama_1b_conformal.slurm
sbatch slurm/v10_llama_3b_conformal.slurm
sbatch slurm/v10_gemma_1b_conformal.slurm
sbatch slurm/v10_gemma_4b_conformal.slurm
sbatch slurm/v10_phi_1.5_conformal.slurm
sbatch slurm/v10_phi_2_conformal.slurm
echo "All 6 jobs submitted to gpu_7day. Results -> results/v10_exact_crc_medquad/"
echo "Compare against results/v9_night_medquad/ once complete."
