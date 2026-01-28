
import json
import numpy as np
import os

def analyze_autopsy(file_path="conformal_autopsy.json"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"--- ANALYZING {file_path} ---")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Data is likely a list of dicts, or the structure from run_eval "results"
    # Based on run_eval.py: it saves a list of records. 
    # Each record has: "initial_answer", "purified_answer", "claims", "scores", "kept_mask"
    
    if isinstance(data, dict) and "records" in data:
         records = data["records"] # Handle if wrapped
    else:
         records = data

    print(f"Total Records: {len(records)}")

    # 1. Retention Rate
    total_claims = 0
    kept_claims = 0
    
    # 2. Score Distribution
    all_scores = []
    
    # 3. Reconstruction Impact
    changed_answers = 0
    
    for r in records:
        # Some records might be failed/empty
        if "claims" not in r or not r["claims"]:
            continue
            
        c_len = len(r["claims"])
        total_claims += c_len
        
        # safely get scores
        s = r.get("scores", [])
        all_scores.extend(s)
        
        # safely get mask
        mask = r.get("kept_mask", [True]*c_len)
        kept = sum(1 for m in mask if m)
        kept_claims += kept
        
        # Check if answer changed
        if r.get("initial_answer") != r.get("purified_answer"):
            changed_answers += 1

    retention_rate = kept_claims / total_claims if total_claims > 0 else 0
    print(f"\nMetric 1: Retention Rate")
    print(f"  Claims Kept: {kept_claims} / {total_claims}")
    print(f"  Rate: {retention_rate:.4f} ({retention_rate*100:.2f}%)")
    
    print(f"\nMetric 2: Score Distribution (Where are the scores?)")
    if all_scores:
        hist, bins = np.histogram(all_scores, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
        for i in range(len(hist)):
            print(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}]: {hist[i]} claims")
    else:
        print("  No scores found.")
        
    print(f"\nMetric 3: Reconstruction Impact")
    print(f"  Answers Modified: {changed_answers} / {len(records)}")
    if changed_answers == 0 and retention_rate < 1.0:
        print("  WARNING: Claims were dropped but Answer text didn't change! (Reconstruction Bug)")

if __name__ == "__main__":
    analyze_autopsy()
