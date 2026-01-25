from datasets import load_dataset
import random
import csv
from pathlib import Path

# MedQuadQ/A dataset from HF
def load_data_local(n: int = 500, seed: int =7, csv_path: str = "data/medDataset_processed.csv"):
    """
    Load the MedQuadQ/A dataset from HF and sample n rows.
    """
    
    # Determine file type
    path_obj = Path(csv_path)
    if not path_obj.exists():
         raise FileNotFoundError(f"Could not find dataset at {path_obj.resolve()}")

    rows = []
    
    # CASE 1: CSV (MedQuad)
    if path_obj.suffix == ".csv":
        with path_obj.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Normalize to standard keys
                rows.append({
                    "question": r["Question"],
                    "answer": r["Answer"],
                    "source": "medquad",
                    "meta": r # Store all other cols
                })

    # CASE 2: JSON (ExpertQA-style)
    elif path_obj.suffix in [".json", ".jsonl"]:
        import json
        with path_obj.open(encoding="utf-8") as f:
            # Handle if it's a list of dicts (json) or lines (jsonl)
            try:
                # Try loading whole file
                raw_data = json.load(f)
                if isinstance(raw_data, list):
                    iterable = raw_data
                else: 
                     iterable = [raw_data] # Single obj
            except json.JSONDecodeError:
                # If fail, try JSONL
                f.seek(0)
                iterable = [json.loads(line) for line in f if line.strip()]
            
            for r in iterable:
                # Adapter: Map "Free_form_answer" -> "answer"
                rows.append({
                    "question": r.get("Question", r.get("question")),
                    "answer": r.get("Free_form_answer", r.get("answer")), 
                    "source": "expertqa",
                    "ground_truth_claims": r.get("Must_have", []),
                    "meta": r
                })
    
    total = len(rows)

    # Sample n values
    if n and n < total:
        print(f"Sampling {n} rows from {path_obj.name} (total={total})")
        rng = random.Random(seed)
        rows = rng.sample(rows, k=n)
    else:
        print(f"Using all {total} rows from {path_obj.name}")
    
    print(f"Loaded {len(rows)} rows.")
    return rows


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

