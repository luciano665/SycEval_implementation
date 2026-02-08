#!/bin/bash

# Submit Llama Jobs
echo "Submitting Llama Family Jobs..."
sbatch slurm/run_experiment_meta_llama.slurm
sbatch slurm/run_conformal_llama_3B.slurm
sbatch slurm/run_conformal_llama_1B.slurm

# Submit Gemma Jobs
echo "Submitting Gemma Family Jobs..."
sbatch slurm/run_experiment_gemma.slurm
sbatch slurm/run_conformal_gemma_4B.slurm
sbatch slurm/run_conformal_gemma_1B.slurm

# Submit Nvidia Jobs
echo "Submitting Nvidia Family Jobs..."
sbatch slurm/run_experiment_nvidia.slurm
sbatch slurm/run_conformal_nvidia_3B.slurm
sbatch slurm/run_conformal_nvidia_1B.slurm

echo "All 9 jobs submitted! Check status with 'squeue -u <username>'."
