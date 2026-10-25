#!/bin/bash
#SBATCH --job-name=rescore_case_claims
#SBATCH --output=logs/output_rescore_case_claims_%j.txt
#SBATCH --error=logs/error_rescore_case_claims_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --partition=gpu_7day

# Recover per-claim scores for the two case-study drafts in fig:cases
# (see docs/rescore_case_claims.py). Read-only on the final N=1000 results;
# writes results/case_claims.json.
set -e
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
export HF_HOME=$PWD/.hf_home
mkdir -p $HF_HOME logs

srun python docs/rescore_case_claims.py --out results/case_claims.json
echo "--- rescore_case_claims exited 0 ---"
