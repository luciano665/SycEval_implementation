# Instructions for WVU HPC Login

```bash
ssh [studentID]@ssh.wvu.edu
```

```bash
ssh [studentID]@ds.hpc.wvu.edu
```

# Instructions for WVU HPC Environment Setup

After Logging into the login node of the HPC (ds.hpc.wvu.edu) run the following commands to setup the environment:

```bash
source /shared/software/conda/etc/profile.d/conda.sh
conda create --name syceval python=3.10 -y
conda activate syceval
```

```bash
cd /path/to/your/SycEval_implementation
pip install -r requirements.txt
```

To run compute nodes of the HPC (standard runs), use the 2-day partition:
```bash
srun -p gpu_2day -N 1 -n 2 --gpus=1 -t 2-00:00:00 --pty /bin/bash
```

To run compute nodes of the HPC (long runs like Conformal), use the 7-day partition:
```bash
srun -p gpu_7day -N 1 -n 2 --gpus=1 -t 7-00:00:00 --pty /bin/bash
```

```bash
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
```

# Deployment: HealthSearchQA Expansion (March 2026)

To run the new HealthSearchQA experiments, follow these steps to pull the latest branch and launch the suite:

### 1. Update Codebase
On the HPC login node (`ds.hpc.wvu.edu`), navigate to your repository and pull the new branch:
```bash
cd /scratch/al00113/SycEval_implementation
git fetch origin
git checkout conformal_v6_healthsearch
git pull origin conformal_v6_healthsearch
```

### 2. Launch Suite
Once on the correct branch, verify the environment and submit the jobs:
```bash
# Ensure conda matches
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval

# Submit all 12 jobs (v6 suite)
chmod +x slurm/submit_v6_suite.sh
./slurm/submit_v6_suite.sh
```

### 3. Monitor Results
Results will accumulate in `results/final_experiment_v6/`. Use `squeue -u al00113` to monitor job status.
