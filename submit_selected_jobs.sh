#!/bin/bash
# (Final Experiment Suite: Feb 21 - Qwen Judge/Rebuttal)

# Function to submit jobs
submit_job() {
    local script_name=$1
    local target_script="slurm/$script_name"
    
    if [ ! -f "$target_script" ]; then
        echo "❌ Error: Could not find $target_script"
        return 1
    fi
    
    # Submit job
    job_output=$(sbatch "$target_script")
    
    # Extract job ID
    job_id=$(echo "$job_output" | awk '{print $4}')
    echo "Submitted batch job $job_id (file: $script_name)"
}

echo "=== Submitting Final Experiment Suite (1000 items, Qwen Judge, 7-Day) ==="

# Llama family
echo "--- Llama ---"
submit_job "suite_b_llama_1b_baseline.slurm"
submit_job "run_experiment_meta_llama.slurm"
submit_job "run_conformal_llama_1B.slurm"
submit_job "run_conformal_llama_3B.slurm"

# Gemma family
echo "--- Gemma ---"
submit_job "suite_b_gemma_1b_baseline.slurm"
submit_job "run_experiment_gemma.slurm"
submit_job "run_conformal_gemma_1B.slurm"
submit_job "run_conformal_gemma_4B.slurm"

# Phi family
echo "--- Phi ---"
submit_job "suite_b_phi_1.5_baseline.slurm"
submit_job "suite_b_phi_2_baseline.slurm"
submit_job "suite_b_phi_1.5_conformal.slurm"
submit_job "suite_b_phi_2_conformal.slurm"

echo "✅ All 12 jobs submitted to gpu_7day partition!"
