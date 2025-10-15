from re import split
from datasets import load_dataset
import random


# MedQuadQ/A dataset from HF
def load_medquad(n=500, seed=7):
    """
    Load the MedQuadQ/A dataset from HF and sample n rows.
    """
    
    dataset = load_dataset("", split="train")

    rows = list(dataset)

    if n and n < len(rows):
        print(f"Sampling {n} rows from MedQuadQ/A dataset")
        rng = random.Random(seed)

        rows = rng.sample(rows, k=n)
    
    print(f"Loaded {len(rows)} rows from MedQuadQ/A dataset")

    return [{"question": r["question"], "answer": r["answer"]} for r in rows]




