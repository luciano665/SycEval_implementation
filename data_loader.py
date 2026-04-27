from datasets import load_dataset
import random
import csv
from pathlib import Path

from logger_utils import get_logger

logger = get_logger(__name__)

# MedQuadQ/A dataset from HF
def load_data_local(n: int = 500, seed: int = 7, csv_path: str = "data/medDataset_processed.csv"):
    """
    Load the MedQuad (CSV) dataset.
    """
    
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset CSV at {csv_path.resolve()}.")

    logger.info(f"Loading MedQuad dataset: {n} items")

    # 1. Load MedDataset
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_med = list(reader)
        
    if n < len(all_med):
        rng = random.Random(seed)
        rows_med = rng.sample(all_med, k=n)
    else:
        rows_med = all_med
        
    data_med = [{"question": r["Question"], "answer": r["Answer"], "source": "medquad"} for r in rows_med]

    logger.info("Loaded %d MedQuad rows", len(data_med))

    return data_med


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

