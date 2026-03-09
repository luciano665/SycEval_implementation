import json
import csv
import argparse
import pandas as pd
from pathlib import Path

def load_dataset_questions(med_csv, hs_jsonl):
    """Load sets of questions from original datasets for mapping."""
    med_qs = set()
    if Path(med_csv).exists():
        with open(med_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                med_qs.add(row['Question'].strip())
    
    hs_qs = set()
    if Path(hs_jsonl).exists():
        with open(hs_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                hs_qs.add(data['Question'].strip())
                
    return med_qs, hs_qs

def summarize_records(records):
    """Compute sycophancy rates for a list of records."""
    if not records:
        return {"count": 0, "overall_sycophancy": 0.0}
    
    df = pd.DataFrame(records)
    total = len(df)
    syco_count = len(df[df['sycophancy'] != 'none'])
    
    # Break down by context if columns exist
    ic_df = df[df['where'] == 'in-context']
    prem_df = df[df['where'] == 'preemptive']
    
    summary = {
        "count": total,
        "overall_sycophancy_rate": syco_count / total if total > 0 else 0,
        "in_context_rate": len(ic_df[ic_df['sycophancy'] != 'none']) / len(ic_df) if len(ic_df) > 0 else 0,
        "preemptive_rate": len(prem_df[prem_df['sycophancy'] != 'none']) / len(prem_df) if len(prem_df) > 0 else 0
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Split mixed results file into MedQuad and HealthSearchQA results.")
    parser.add_argument("--input", type=str, required=True, help="Path to the mixed result JSON file")
    parser.add_argument("--med_csv", type=str, default="data/medDataset_processed.csv")
    parser.add_argument("--hs_jsonl", type=str, default="data/healthsearch_qa.jsonl")
    
    args = parser.parse_args()
    
    # 1. Load data
    print(f"Loading mapping from {args.med_csv} and {args.hs_jsonl}...")
    med_qs, hs_qs = load_dataset_questions(args.med_csv, args.hs_jsonl)
    
    with open(args.input, 'r', encoding='utf-8') as f:
        full_results = json.load(f)
        
    records = full_results.get("individual_records", [])
    if not records:
        print("No individual_records found in input file.")
        return

    # 2. Split
    med_records = []
    hs_records = []
    unknown = []
    
    for r in records:
        q = r.get("question", "").strip()
        if q in med_qs:
            med_records.append(r)
        elif q in hs_qs:
            hs_records.append(r)
        else:
            unknown.append(r)
            
    print(f"Split results: MedQuad({len(med_records)}), HealthSearch({len(hs_records)}), Unknown({len(unknown)})")
    
    # 3. Summarize and Save
    base_name = Path(args.input).stem
    
    for label, subset in [("medquad", med_records), ("healthsearch", hs_records)]:
        if not subset:
            continue
            
        summary = summarize_records(subset)
        out_path = f"results/split_{base_name}_{label}.json"
        
        output = {
            "metadata": {
                "original_file": args.input,
                "dataset_type": label,
                "summary": summary
            },
            "records": subset
        }
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        print(f"Saved {label} results to {out_path}")
        print(f"  -> {label.upper()} Sycophancy Rate: {summary['overall_sycophancy_rate']:.2%}")

if __name__ == "__main__":
    main()
