
import json
import pandas as pd
import os
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# --------------------------
# CONFIG
# --------------------------
input_files = [
    "results/gemma_distill_results_1000.json",
    "results/llama3_2_distill_results_1000.json",
    "results/nvidia_distill_results_1000.json"
]

def load_data():
    all_records = []
    print("Loading data for statistical analysis...")
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}")
            continue
            
        fname = os.path.basename(file_path).lower()
        if "gemma" in fname: family = "Gemma"
        elif "llama" in fname: family = "Llama"
        elif "nvidia" in fname: family = "Nvidia"
        else: family = "Other"
        
        with open(file_path, "r") as f:
            data = json.load(f)
            recs = data["records"]
            for r in recs:
                r["family"] = family
                if "teacher" in r["model"].lower():
                    r["role"] = "Teacher"
                elif "student" in r["model"].lower():
                    r["role"] = "Student"
                else:
                    r["role"] = "Unknown"
            all_records.extend(recs)

    df = pd.DataFrame(all_records)
    
    # Flags
    df["first_correct"] = df["first_label"] == "correct"
    df["after_correct"] = df["after_label"] == "correct"
    df["is_sycophantic"] = df["sycophancy"] != "none"
    
    return df

def run_mcnemar(df, condition1_col, condition2_col, label):
    """
    Runs McNemar's test on two binary columns.
    Null hypothesis: Marginal frequencies are equal (no significant difference).
    """
    if len(df) == 0:
        return None
        
    # Contingency Table
    #           Cond2 False | Cond2 True
    # Cond1 False    a      |      b
    # Cond1 True     c      |      d
    
    # We are interested in the discordance (b vs c).
    # b: Cond1=False, Cond2=True
    # c: Cond1=True, Cond2=False
    
    # Use crosstab
    ct = pd.crosstab(df[condition1_col], df[condition2_col])
    
    # Ensure 2x2
    if ct.shape != (2, 2):
        # Handle edge cases where one condition is constant
        return {"p_value": np.nan, "significant": False, "note": "Insufficient variance"}

    # statsmodels mcnemar
    # It expects: [[a, b], [c, d]]
    # a: 0,0 | b: 0,1
    # c: 1,0 | d: 1,1
    
    # Note: pd.crosstab default sorts (False, True). 
    # So [0,0] is False/False.
    
    result = mcnemar(ct, exact=True) # Exact test is safer for small counts, though we have ~1000
    
    return {
        "p_value": result.pvalue,
        "significant": result.pvalue < 0.05,
        "statistic": result.statistic
    }

def analyze_before_after(df):
    print("\n--- Test 1: Impact of Intervention (Before vs After Accuracy) ---")
    results = []
    
    # Group by Model (Family + Role)
    for (family, role), group in df.groupby(["family", "role"]):
        if role == "Unknown": continue
        
        # McNemar: Compare 'first_correct' vs 'after_correct' on the SAME rows
        res = run_mcnemar(group, "first_correct", "after_correct", "Intervention Impact")
        
        if res:
            results.append({
                "Family": family,
                "Role": role,
                "Comparison": "Before vs After",
                "P-Value": res["p_value"],
                "Significant": "YES" if res["significant"] else "No"
            })
            
    return pd.DataFrame(results)

def analyze_teacher_student(df):
    print("\n--- Test 2 & 3: Teacher vs Student (Paired) ---")
    results = []
    
    # We need to pivot to pair them up
    # Pivot on [qid, bucket, mode, strength, run_id]
    # We need strictly unique pairs.
    
    for family in ["Gemma", "Llama", "Nvidia"]:
        fam_df = df[df["family"] == family]
        if fam_df.empty: continue
        
        # Separate Teacher and Student
        teacher_df = fam_df[fam_df["role"] == "Teacher"]
        student_df = fam_df[fam_df["role"] == "Student"]
        
        # Merge on ID columns
        merged = pd.merge(
            teacher_df, 
            student_df, 
            on=["qid", "run_id"], # minimal key, maybe add mode/strength if needed for uniqueness
            suffixes=("_T", "_S")
        )
        
        if merged.empty:
            print(f"Skipping {family} (No pairs found)")
            continue
            
        # Test 2: Sycophancy (Teacher vs Student)
        # Compare is_sycophantic_T vs is_sycophantic_S
        res_syc = run_mcnemar(merged, "is_sycophantic_T", "is_sycophantic_S", "Sycophancy")
        if res_syc:
            results.append({
                "Family": family,
                "Comparison": "Sycophancy Rate (T vs S)",
                "P-Value": res_syc["p_value"],
                "Significant": "YES" if res_syc["significant"] else "No"
            })
            
        # Test 3: Accuracy After Intervention
        # Compare after_correct_T vs after_correct_S
        res_acc = run_mcnemar(merged, "after_correct_T", "after_correct_S", "Accuracy After")
        if res_acc:
             results.append({
                "Family": family,
                "Comparison": "Accuracy After (T vs S)",
                "P-Value": res_acc["p_value"],
                "Significant": "YES" if res_acc["significant"] else "No"
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    try:
        df = load_data()
        if df.empty:
            print("No data found.")
            exit()
            
        # 1. Before vs After
        res_ba = analyze_before_after(df)
        print("\n[Before vs After Accuracy]")
        print(res_ba.to_string(index=False))
        
        # 2. Teacher vs Student
        res_ts = analyze_teacher_student(df)
        print("\n[Teacher vs Student Comparisons]")
        print(res_ts.to_string(index=False))
        
        # Save to csv
        res_ba.to_csv("stats_before_after.csv", index=False)
        res_ts.to_csv("stats_teacher_student.csv", index=False)
        print("\nSaved statistics to CSV files.")
        
    except ImportError:
        print("Error: statsmodels is required. Please install it with: pip install statsmodels")
