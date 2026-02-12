#!/bin/bash
# Master Submission Script for Qwen Neutral experiments (Ministral-8B Judge/Rebuttal)

# Function to submit jobs
submit_job() {
    local script_path=$1
    if [ -f "$script_path" ]; then
        local job_output=$(sbatch "$script_path")
        local job_id=$(echo "$job_output" | awk '{print $4}')
        echo "Submitted batch job $job_id (path: $script_path)"
    else
        echo "❌ Error: Could not find $script_path"
    fi
}

echo "=== Submitting Neutral Qwen Suite (Judge/Rebuttal: Ministral-8B) ==="

# Qwen Family
echo "--- Qwen ---"
submit_job "slurm/neutral_qwen_baseline.slurm"
submit_job "slurm/neutral_qwen_1.5B_conformal.slurm"
submit_job "slurm/neutral_qwen_3B_conformal.slurm"

echo "✅ All 3 neutral Qwen jobs submitted!"
