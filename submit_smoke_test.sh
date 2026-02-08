#!/bin/bash
mkdir -p logs

echo "Submitting Smoke Tests (100 items each)..."
sbatch slurm/test_smoke_llama_baseline.slurm
sbatch slurm/test_smoke_llama_conformal.slurm

echo "Smoke tests submitted! Check status with 'squeue -u <username>'."
echo "Results will be saved to results/smoke_test_v1/"
