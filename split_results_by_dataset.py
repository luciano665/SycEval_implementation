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
        return {"count": 0}
    
    df = pd.DataFrame(records)
    total = len(df)
    
    # Baseline (Draft) Sycophancy
    # Use 'draft_sycophancy' if available (Conformal), else use 'sycophancy' (Baseline file)
    baseline_col = 'draft_sycophancy' if 'draft_sycophancy' in df.columns else 'sycophancy'
    baseline_syco_count = len(df[df[baseline_col] != 'none'])
    
    # Final (Filtered/Rewritten) Sycophancy 
    # For a Baseline file, final same as baseline
    final_syco_count = len(df[df['sycophancy'] != 'none'])
    
    # Break down by context
    ic_df = df[df['mode'] == 'in-context'] if 'mode' in df.columns else df[df['where'] == 'in-context']
    prem_df = df[df['mode'] == 'preemptive'] if 'mode' in df.columns else df[df['where'] == 'preemptive']
    
    def get_rate(subset, col):
        if len(subset) == 0: return 0.0
        return len(subset[subset[col] != 'none']) / len(subset)

    summary = {
        "count": total,
        "baseline_overall_rate": baseline_syco_count / total if total > 0 else 0,
        "final_overall_rate": final_syco_count / total if total > 0 else 0,
        "sycophancy_delta": (final_syco_count - baseline_syco_count) / total if total > 0 else 0,
        "in_context": {
            "baseline": get_rate(ic_df, baseline_col),
            "final": get_rate(ic_df, 'sycophancy')
        },
        "preemptive": {
            "baseline": get_rate(prem_df, baseline_col),
            "final": get_rate(prem_df, 'sycophancy')
        }
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
        
    # Handle both run_eval.py (records) and run_conformal_v2.py (individual_records)
    records = full_results.get("individual_records") or full_results.get("records")
    if not records:
        print(f"No records found in {args.input}. Keys: {list(full_results.keys())}")
        return

    # 2. Split
    med_records = []
    hs_records = []
    unknown = []
    
    for r in records:
        # Check both common keys for question
        q = r.get("question") or r.get("Question") or ""
        q = q.strip()
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
        print(f"  -> {label.upper()} Baseline Rate: {summary['baseline_overall_rate']:.2%}")
        print(f"  -> {label.upper()} Final Rate:    {summary['final_overall_rate']:.2%}")
        print(f"  -> {label.upper()} Delta:         {summary['sycophancy_delta']:+.2%}")

if __name__ == "__main__":
    main()
