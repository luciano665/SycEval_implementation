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

To run compute nodes of the HPC, run the following command:

```bash
srun -p inter_a30 -N 1 -n 2 --gpus=1 -t 00:30:00 --pty /bin/bash
```

```bash
source /shared/software/conda/etc/profile.d/conda.sh
conda activate syceval
```

