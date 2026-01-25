
import os
import subprocess

# Define the grid
datasets = [
    {"name": "medquad", "path": "data/medDataset_processed.csv"},
    {"name": "healthsearch", "path": "data/healthsearch_qa.jsonl"}
]

# Using the "Student" models from the pairs as the Tested Models
models = [
    {"name": "llama3b", "path": "/users/al00113/models/Llama-3.2-3B-Instruct"},
    {"name": "llama8b", "path": "/users/al00113/models/Llama-3.1-8B-Instruct"},
    {"name": "gemma4b", "path": "/users/al00113/models/gemma-3-4b-it"},
    {"name": "nemotron3b", "path": "/users/al00113/models/Nemotron-Flash-3B"},
]

slurm_script = "slurm/run_conformal.slurm"
threshold = 0.8  # Change this if calibration implies a different value

print(f"--- Launching Parallel Experiments (Threshold={threshold}) ---")

for mod in models:
    for ds in datasets:
        
        job_name = f"eval_{mod['name']}_{ds['name']}"
        out_file = f"results/{job_name}.json"
        
        # Construct sbatch command with exports
        # We export TESTED_MODEL, DATASET_PATH, OUT_FILE, CONFORMAL_THRESHOLD to the SLURM script
        cmd = [
            "sbatch",
            f"--job-name={job_name}",
            f"--output=logs/{job_name}_%j.txt",
            f"--export=ALL,TESTED_MODEL={mod['path']},DATASET_PATH={ds['path']},OUT_FILE={out_file},CONFORMAL_THRESHOLD={threshold}",
            slurm_script
        ]
        
        print(f"Submitting: {job_name}")
        # print(" ".join(cmd))
        subprocess.run(cmd)

print("--- All jobs submitted ---")
