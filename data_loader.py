from datasets import load_dataset
import random
import csv
from pathlib import Path

from logger_utils import get_logger

logger = get_logger(__name__)

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

    import json
    
    rows = []
    
    if csv_path.suffix == ".jsonl":
        with csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        # Assume CSV
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

    total = len(rows)

    # Sample n values from CSV file
    if n and n < total:
        logger.info("Sampling %d rows from %s (total=%d)", n, csv_path.name, total)
        rng = random.Random(seed)
        rows = rng.sample(rows, k=n)

    else:
        logger.info("Using all %d rows from %s", total, csv_path.name)
    
    logger.info(
        "Loaded %d rows from %s",
        len(rows),
        csv_path.name
    )

    # Helper to find keys case-insensitively
    def get_val(r, keys):
        for k in keys:
            if k in r: return r[k]
        return ""

    # Return on format {"question": r["Question"], "answer": r["Answer"]}
    return [{"question": get_val(r, ["Question", "question"]), "answer": get_val(r, ["Answer", "answer"])} for r in rows]


# MedQuadQ/A laoder funtion to use not on HPC
def load_data(n=500, seed=7):
    """
    Load the MedQuadQ/A dataset from HF and sample n rows.
    """
    
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")

    rows = list(dataset)

    if n and n < len(rows):
        logger.info("Sampling %d rows from MedQuadQ/A dataset", n)
        rng = random.Random(seed)

        rows = rng.sample(rows, k=n)
    
    logger.info("Loaded %d rows from MedQuadQ/A dataset", len(rows))

    return [{"question": r["Question"], "answer": r["Answer"]} for r in rows]

