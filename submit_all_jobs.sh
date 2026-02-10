#!/bin/bash

# 1. Download Qwen, Mistral Judge, and other models
echo "=== Submitting Model Download Job ==="
JOB_ID=$(sbatch slurm/download_models_qwen_mistral.slurm | awk '{print $4}')
echo "Download Job ID: $JOB_ID"

# 2. Submit Experiment Jobs (All dependent on download)

# Qwen Experiments (New)
echo "=== Submitting Qwen Experiments (Mistral Judge, 48h) ==="
sbatch --dependency=afterok:$JOB_ID slurm/run_experiment_qwen.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_qwen_1.5B.slurm
sbatch --dependency=afterok:$JOB_ID slurm/run_conformal_qwen_3B.slurm

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

echo "✅ All 9 jobs submitted! Experiments will start after models are downloaded."
