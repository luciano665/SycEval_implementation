from datasets import load_dataset
import random
import csv
import json
from pathlib import Path

from logger_utils import get_logger

logger = get_logger(__name__)

def load_medquad(n: int = 500, seed: int = 7, csv_path: str = "data/medDataset_processed.csv"):
    """
    Load the MedQuad (CSV) dataset.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find MedQuad dataset at {csv_path.resolve()}.")

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_data = list(reader)
        
    if n < len(all_data):
        rng = random.Random(seed)
        rows = rng.sample(all_data, k=n)
    else:
        rows = all_data
        
    return [{"question": r["Question"], "answer": r["Answer"], "source": "medquad"} for r in rows]

def load_healthsearch(n: int = 500, seed: int = 7, jsonl_path: str = "data/healthsearch_qa.jsonl"):
    """
    Load the HealthSearchQA (JSONL) dataset.
    Uses Free_form_answer as the ground truth primary answer.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Could not find HealthSearchQA dataset at {jsonl_path.resolve()}.")

    all_data = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))
            
    if n < len(all_data):
        rng = random.Random(seed)
        rows = rng.sample(all_data, k=n)
    else:
        rows = all_data
        
    return [{"question": r["Question"], "answer": r["Free_form_answer"], "source": "healthsearch"} for r in rows]

def load_data_local(n: int = 500, seed: int = 7, domain: str = "medquad"):
    """
    Unified local loader for different domains.
    """
    logger.info(f"Loading local dataset: {domain} ({n} items)")
    
    if domain == "medquad":
        return load_medquad(n, seed)
    elif domain == "healthsearch":
        return load_healthsearch(n, seed)
    else:
        raise ValueError(f"Unknown domain: {domain}")


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
