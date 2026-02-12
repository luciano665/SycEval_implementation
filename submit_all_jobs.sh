#!/bin/bash
# (HPC Sync Check: Feb 12 - NO DOWNLOAD TEST)

# Function to submit jobs (Robust path check)
submit_job() {
    local script_name=$1
    local target_script=""
    local job_output=""
    local job_id=""

    # Check for file existence in current directory or slurm/ subdirectory
    if [ -f "./$script_name" ]; then
        target_script="./$script_name"
    elif [ -f "slurm/$script_name" ]; then
        target_script="slurm/$script_name"
    else
        echo "❌ Error: Could not find $script_name in . or slurm/"
        return 1
    fi
    
    # Submit job (NO DEPENDENCY)
    job_output=$(sbatch "$target_script")
    
    # Extract job ID
    job_id=$(echo "$job_output" | awk '{print $4}')
    echo "Submitted batch job $job_id (path: $target_script)"
}

echo "=== Submitting Experiment Jobs (Skipping Download - Models Pre-Installed) ==="

# Llama Experiments
echo "--- Llama ---"
submit_job "run_experiment_meta_llama.slurm"
submit_job "run_conformal_llama_1B.slurm"
submit_job "run_conformal_llama_3B.slurm"

# Gemma Experiments
echo "--- Gemma ---"
submit_job "run_experiment_gemma.slurm"
submit_job "run_conformal_gemma_1B.slurm"
submit_job "run_conformal_gemma_4B.slurm"

# Nvidia Experiments
echo "--- Nvidia ---"
submit_job "run_experiment_nvidia.slurm"
submit_job "run_conformal_nvidia_1B.slurm"
submit_job "run_conformal_nvidia_3B.slurm"

echo "✅ All 9 jobs submitted! Running immediately."
