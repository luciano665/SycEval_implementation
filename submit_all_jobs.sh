#!/bin/bash
# (HPC Sync Check: Feb 12)


# 1. Download Qwen, Mistral Judge, and other models
echo "=== Submitting Model Download Job ==="
JOB_ID=$(sbatch slurm/download_models_qwen_mistral.slurm | awk '{print $4}')
echo "Download Job ID: $JOB_ID"

# 2. Submit Experiment Jobs (All dependent on download)

# Qwen Experiments (New)
# Qwen Experiments (Skipped)
# echo "=== Submitting Qwen Experiments (Qwen Judge, 24h) ==="
# sbatch --dependency=afterok:$JOB_ID slurm/run_experiment_qwen.slurm
# sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_qwen_1.5B.slurm
# sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_qwen_3B.slurm

# Llama Experiments (Rerunning with Mistral Judge for consistency)
echo "=== Submitting Llama Experiments (Mistral Judge, 48h) ==="
sbatch --dependency=afterok:$JOB_ID slurm/run_experiment_meta_llama.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_llama_1B.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_llama_3B.slurm

# Gemma Experiments (Rerunning with Mistral Judge for consistency)
echo "=== Submitting Gemma Experiments (Mistral Judge, 48h) ==="
sbatch --dependency=afterok:$JOB_ID slurm/run_experiment_gemma.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_gemma_1B.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_gemma_4B.slurm

# Nvidia Experiments (Local Weights, Qwen Judge)
echo "=== Submitting Nvidia Experiments (Qwen Judge, 24h) ==="
sbatch --dependency=afterok:$JOB_ID slurm/run_experiment_nvidia.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_nvidia_1B.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_nvidia_3B.slurm

echo "✅ All 12 jobs submitted! Experiments will start after models are downloaded."
