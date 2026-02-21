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
