
import json
import numpy as np
import os

def analyze_autopsy(file_path="conformal_results.json"):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"--- ANALYZING {file_path} ---")
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Data is likely a list of dicts, or the structure from run_eval "results"
    # Based on run_eval.py: it saves a list of records. 
    # Each record has: "initial_answer", "purified_answer", "claims", "scores", "kept_mask"
    
    if isinstance(data, dict):
        if "individual_records" in data:
            records = data["individual_records"]
        elif "records" in data:
            records = data["records"]
        else:
            # Maybe it is flat?
            records = [data] # Unexpected
    else:
        records = data

    print(f"Total Records Found: {len(records)}")

    # 1. Retention Rate
    total_claims = 0
    total_dropped = 0
    
    # 2. Score Distribution (Might not be in this file if lightweight)
    all_scores = []
    
    # 3. Reconstruction Impact
    changed_answers = 0
    
    for r in records:
        # Get counts directly if available
        if "conformal_dropped_count" in r:
            drop_c = r.get("conformal_dropped_count", 0)
            orig_c = r.get("conformal_original_count", 0)
            total_dropped += drop_c
            total_claims += orig_c
        
        # safely get mask
        # If mask is present, we can double check
        if "kept_mask" in r:
             # Logic if mask is there
             pass
        
        # Check if answer changed
        # We might not have full text in this lightweight file?
        if "initial_answer" in r and "purified_answer" in r:
            if r["initial_answer"] != r["purified_answer"]:
                changed_answers += 1
        else:
             # Just assume no change if text missing
             pass

    retention_rate = (total_claims - total_dropped) / total_claims if total_claims > 0 else 0
    print(f"\nMetric 1: Global Retention Rate")
    print(f"  Total Claims: {total_claims}")
    print(f"  Total Dropped: {total_dropped}")
    print(f"  Claims Kept: {total_claims - total_dropped}")
    print(f"  Retention Rate: {retention_rate:.4f} ({retention_rate*100:.2f}%)")
    
    if retention_rate < 0.1:
        print("  CRITICAL: We are dropping >90% of claims! This is too aggressive.")
    elif retention_rate > 0.9:
        print("  WARNING: We are keeping >90% of claims. Filter does nothing.")

    print(f"\nMetric 3: Reconstruction Impact")
    if changed_answers > 0:
        print(f"  Answers Modified: {changed_answers} / {len(records)}")
    else:
        print("  Note: 'initial_answer' and 'purified_answer' text not found in summary JSON.")
        print("  Cannot verify text changes directly from this file.")

if __name__ == "__main__":
    analyze_autopsy()
