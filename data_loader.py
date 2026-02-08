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

    # Determine split (50/50 if n is uniform, otherwise maintain request)
    # User requested 500 from each if n=1000
    n_med = n // 2
    n_hs = n - n_med
    
    logger.info(f"Loading mixed dataset: {n_med} from MedQuad, {n_hs} from HealthSearch")

    # 1. Load MedDataset
    rows_med = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_med = list(reader)
        
    if n_med < len(all_med):
        rng = random.Random(seed)
        rows_med = rng.sample(all_med, k=n_med)
    else:
        rows_med = all_med
        
    data_med = [{"question": r["Question"], "answer": r["Answer"]} for r in rows_med]

    # 2. Load HealthSearch
    hs_path = Path("data/healthsearch_qa.jsonl")
    data_hs = []
    if hs_path.exists():
        import json
        with hs_path.open("r", encoding="utf-8") as f:
            all_hs = [json.loads(line) for line in f]
            
        if n_hs < len(all_hs):
            rng = random.Random(seed + 1) # Different seed for variety
            rows_hs = rng.sample(all_hs, k=n_hs)
        else:
            rows_hs = all_hs
            
        # Process HealthSearch: Answer is joined "Must_have" list
        for r in rows_hs:
            ans_str = " ".join(r.get("Must_have", []))
            data_hs.append({"question": r["Question"], "answer": ans_str})
    else:
        logger.warning(f"HealthSearch dataset not found at {hs_path}. Only using MedDataset.")

    # Combine and Shuffle
    combined_data = data_med + data_hs
    random.Random(seed).shuffle(combined_data)
    
    logger.info(
        "Loaded %d total rows (%d MedQuad, %d HealthSearch)",
        len(combined_data), len(data_med), len(data_hs)
    )

    return combined_data


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

