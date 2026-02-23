#!/bin/bash
# 🚀 Submit only the jobs that failed due to the directory mismatch.
# (Running jobs 100044, 100043, 100042 are excluded)

submit_job() {
    local script_name=$1
    local target_script="slurm/$script_name"
    if [ ! -f "$target_script" ]; then
        echo "❌ Error: Could not find $target_script"
        return 1
    fi
    job_output=$(sbatch "$target_script")
    job_id=$(echo "$job_output" | awk '{print $4}')
    echo "Re-submitted batch job $job_id (file: $script_name)"
}

echo "=== Re-submitting Failed Experiments Only ==="

# Llama family (All failed)
echo "--- Llama ---"
submit_job "suite_b_llama_1b_baseline.slurm"
submit_job "run_experiment_meta_llama.slurm"
submit_job "run_conformal_llama_1B.slurm"
submit_job "run_conformal_llama_3B.slurm"

# Gemma family (Baseline & 1B Conformal failed)
echo "--- Gemma ---"
submit_job "suite_b_gemma_1b_baseline.slurm"
submit_job "run_experiment_gemma.slurm"
submit_job "run_conformal_gemma_1B.slurm"
# (Skipping run_conformal_gemma_4B.slurm - it is currently running as job 100042)

# Phi family (Baselines failed)
echo "--- Phi ---"
submit_job "suite_b_phi_1.5_baseline.slurm"
submit_job "suite_b_phi_2_baseline.slurm"
# (Skipping conformal jobs - they are running as 100043 and 100044)

echo "✅ Failed jobs re-submitted!"
