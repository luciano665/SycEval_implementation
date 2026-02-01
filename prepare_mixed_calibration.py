
import csv
import json
import random
import pandas as pd
from pathlib import Path

def prepare_mixed_calibration(
    med_path="data/medDataset_processed.csv",
    hs_path="data/healthsearch_qa.jsonl",
    output_path="data/calibration_mixed.jsonl",
    n_per_source=150,
    seed=42
):
    print(f"Preparing mixed calibration dataset (n={n_per_source} per source, seed={seed})...")
    random.seed(seed)

    mixed_data = []

    # 1. Load MedQuad (CSV)
    # Expected columns: Question, Answer
    print(f"Loading MedQuad from {med_path}...")
    try:
        df_med = pd.read_csv(med_path)
        # Check columns
        if 'Question' not in df_med.columns or 'Answer' not in df_med.columns:
            # Fallback for lowercase
            if 'question' in df_med.columns and 'answer' in df_med.columns:
                df_med = df_med.rename(columns={'question': 'Question', 'answer': 'Answer'})
            else:
                raise ValueError(f"MedQuad CSV missing 'Question'/'Answer' columns. Found: {df_med.columns}")
        
        med_records = df_med.to_dict('records')
        print(f"  Found {len(med_records)} MedQuad records.")
        
        # Sample
        if len(med_records) > n_per_source:
            med_sample = random.sample(med_records, n_per_source)
        else:
            med_sample = med_records
            print(f"  Warning: Only {len(med_records)} available, taking all.")

        # Format
        for r in med_sample:
            mixed_data.append({
                "question": r['Question'],
                "answer": r['Answer'],
                "source": "medquad",
                "original_id": str(r.get('qtype', '')) # Optional meta
            })
            
    except Exception as e:
        print(f"Error loading MedQuad: {e}")
        return

    # 2. Load HealthSearchQA (JSONL)
    # Expected keys: Question, Must_have (list)
    print(f"Loading HealthSearchQA from {hs_path}...")
    hs_records = []
    try:
        with open(hs_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    hs_records.append(json.loads(line))
        
        print(f"  Found {len(hs_records)} HealthSearchQA records.")
        
        # Sample
        if len(hs_records) > n_per_source:
            hs_sample = random.sample(hs_records, n_per_source)
        else:
            hs_sample = hs_records
            print(f"  Warning: Only {len(hs_records)} available, taking all.")

        # Format
        for r in hs_sample:
            # Join 'Must_have' list into a single string for 'answer'
            must_have = r.get('Must_have', [])
            if isinstance(must_have, list):
                answer_text = " ".join(must_have)
            else:
                answer_text = str(must_have)
            
            mixed_data.append({
                "question": r['Question'],
                "answer": answer_text,
                "source": "healthsearch_qa",
                "original_meta": {k:v for k,v in r.items() if k not in ['Question', 'Must_have']}
            })

    except Exception as e:
        print(f"Error loading HealthSearchQA: {e}")
        return

    # 3. Save
    print(f"Saving {len(mixed_data)} records to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(json.dumps(item) + "\n")
    
    print("Done.")

if __name__ == "__main__":
    prepare_mixed_calibration()
