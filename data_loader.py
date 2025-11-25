from datasets import load_dataset
import random
import csv
from pathlib import Path

# MedQuadQ/A dataset from HF
def load_data_local(n: int = 500, seed: int =7, csv_path: str = "data/medDataset_processed.csv"):
    """
    Load the MedQuadQ/A dataset from HF and sample n rows.
    """
    
    # Convert to Path object
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset CSV at {csv_path.resolve()}. "
            "Make sure 'medDataset_processed.csv' is placed there."
        )

    rows = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        # Init reading csv
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total = len(rows)

    # Sample n values from CSV file
    if n and n < total:
        print(f"Sampling {n} rows from MedQuad Q/A local CSV (total={total})")
        rng = random.Random(seed)
        rows = rng.sample(rows, k=n)

    else:
        print(f"Using all {total} rows from MedQuad Q/A local CSV")
    
    print(f"Loaded {len(rows)} rows from MedQuad Q/A dataset (local CSV)")

    # Return on format {"question": r["Question"], "answer": r["Answer"]}
    return [{"question": r["Question"], "answer": r["Answer"]} for r in rows]


# MedQuadQ/A laoder funtion to use not on HPC
def load_data(n=500, seed=7):
    """
    Load the MedQuadQ/A dataset from HF and sample n rows.
    """
    
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

    rows = list(dataset)

    if n and n < len(rows):
        print(f"Sampling {n} rows from MedQuadQ/A dataset")
        rng = random.Random(seed)

        rows = rng.sample(rows, k=n)
    
    print(f"Loaded {len(rows)} rows from MedQuadQ/A dataset")

    return [{"question": r["Question"], "answer": r["Answer"]} for r in rows]

