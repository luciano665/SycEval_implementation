from datasets import load_dataset
import random


# MedQuadQ/A dataset from HF
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




