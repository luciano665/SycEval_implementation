#!/bin/bash
#SBATCH --job-name=rescore_case_claims
#SBATCH --output=logs/output_rescore_case_claims_%j.txt
#SBATCH --error=logs/error_rescore_case_claims_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_2day

# Recover per-claim scores for the two case-study drafts in fig:cases
# (see docs/rescore_case_claims.py). Read-only on the final N=1000 results;
# writes results/case_claims_{a,b}.json.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME logs

nvidia-smi || { echo "GPU not usable on $(hostname); resubmit with --exclude=$(hostname)"; exit 1; }
# One process per panel so only one tested model (plus Qwen) is loaded at a time.
srun python docs/rescore_case_claims.py --panel a --out results/case_claims_a.json
srun python docs/rescore_case_claims.py --panel b --out results/case_claims_b.json
echo "--- rescore_case_claims exited 0 ---"
