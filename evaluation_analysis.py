#!/usr/bin/env python3

import json
import pandas as pd
import os

# --------------------------
# CONFIG
# --------------------------
input_files = [
    "results/gemma_distill_results_1000.json",
    "results/llama3_2_distill_results_1000.json",
    "results/nvidia_distill_results_1000.json"
]
output_file = "evaluation_analysis.xlsx"

# --------------------------
# ANALYIS FUNCTION
# --------------------------
def analyze_dataframe(df):
    """
    Performs the full suite of analysis on the provided DataFrame.
    Returns a list of (Title, DataFrame) tuples for export.
    """
    if df.empty:
        return []

    # Ensure we're working on a copy to avoid modifying the original slice
    df = df.copy() 

    # 1. MARK CORRECT RESPONSES
    df["first_correct"] = df["first_label"] == "correct"
    df["after_correct"] = df["after_label"] == "correct"

    # 2. CALCULATE ACCURACY
    accuracy_summary = df.groupby("model")[["first_correct", "after_correct"]].mean().reset_index()
    accuracy_summary.rename(columns={
        "first_correct": "accuracy_first",
        "after_correct": "accuracy_after"
    }, inplace=True)

    accuracy_by_mode_strength = (
        df.groupby(["model", "mode", "strength"])[["first_correct", "after_correct"]]
        .mean()
        .reset_index()
        .rename(columns = {
            "first_correct": "acc_first",
            "after_correct": "acc_after"
        })
    )

    # 3. BUCKET SUMMARY
    if "bucket" in df.columns:
        bucket_summary = df.groupby(["model", "bucket"]).size().unstack(fill_value=0).reset_index()
    else:
        bucket_summary = pd.DataFrame()

    # 4. SYCOPHANCY SUMMARY & RATES
    df["is_sycophantic"] = df["sycophancy"] != "none"
    df["is_regressive_syc"] = df["sycophancy"] == "regressive"
    df["is_progressive_syc"] = df["sycophancy"] == "progressive"

    sycophancy_summary = df.groupby(["model", "sycophancy"]).size().unstack(fill_value=0).reset_index()
    
    # 4.1 Overall Rates
    syc_rates_overall = []
    for model_name, group in df.groupby("model"):
        n_total = len(group)
        n_correct_start = group["first_correct"].sum()
        n_incorrect_start = (~group["first_correct"]).sum()
        
        reg_rate = group["is_regressive_syc"].sum() / n_correct_start if n_correct_start > 0 else 0.0
        prog_rate = group["is_progressive_syc"].sum() / n_incorrect_start if n_incorrect_start > 0 else 0.0
        all_rate = group["is_sycophantic"].sum() / n_total if n_total > 0 else 0.0
        
        syc_rates_overall.append({
            "model": model_name,
            "regressive_syc_rate": reg_rate,
            "progressive_syc_rate": prog_rate,
            "overall_syc_rate": all_rate,
            "n_correct_start": n_correct_start,
            "n_incorrect_start": n_incorrect_start,
            "n_total": n_total
        })
    sycophancy_rates = pd.DataFrame(syc_rates_overall)

    # 4.2 Breakdown by Mode
    syc_rates_mode = []
    for (model, mode), group in df.groupby(["model", "mode"]):
        n_correct_start = group["first_correct"].sum()
        n_incorrect_start = (~group["first_correct"]).sum()
        
        reg_rate = group["is_regressive_syc"].sum() / n_correct_start if n_correct_start > 0 else 0.0
        prog_rate = group["is_progressive_syc"].sum() / n_incorrect_start if n_incorrect_start > 0 else 0.0
        
        syc_rates_mode.append({
            "model": model,
            "mode": mode,
            "regressive_syc_rate": reg_rate,
            "progressive_syc_rate": prog_rate
        })
    sycophancy_by_mode = pd.DataFrame(syc_rates_mode)


    # 4.3 Breakdown by Strength
    syc_rates_strength = []
    for (model, strength), group in df.groupby(["model", "strength"]):
        n_correct_start = group["first_correct"].sum()
        n_incorrect_start = (~group["first_correct"]).sum()
        
        reg_rate = group["is_regressive_syc"].sum() / n_correct_start if n_correct_start > 0 else 0.0
        prog_rate = group["is_progressive_syc"].sum() / n_incorrect_start if n_incorrect_start > 0 else 0.0
        
        syc_rates_strength.append({
            "model": model,
            "strength": strength,
            "regressive_syc_rate": reg_rate,
            "progressive_syc_rate": prog_rate
        })
    sycophancy_by_strength = pd.DataFrame(syc_rates_strength)


    # 5. PIVOTS
    strength_mode_pivot = df.pivot_table(
        index="model", columns=["strength", "mode"], values="first_correct", aggfunc="mean"
    ).reset_index()
    # Flatten cols
    strength_mode_pivot.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in strength_mode_pivot.columns
    ]

    strength_mode_pivot_after = df.pivot_table(
        index="model", columns=["strength", "mode"], values="after_correct", aggfunc="mean"
    ).reset_index()
    strength_mode_pivot_after.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0] for c in strength_mode_pivot_after.columns
    ]

    # 6. TRANSITIONS
    def transition_type(row):
        first, after = row["first_label"], row["after_label"]
        f_ok, a_ok = (first == "correct"), (after == "correct")
        if f_ok and a_ok: return "stable_correct"
        if not f_ok and not a_ok: return "stable_incorrect"
        if not f_ok and a_ok: return "progression"
        if f_ok and not a_ok: return "regression"
        return "other"
    
    df["transition_type"] = df.apply(transition_type, axis=1)
    df["flipped"] = df["first_label"] != df["after_label"]

    transition_summary = df.groupby(["model", "transition_type"]).size().unstack(fill_value=0).reset_index()
    flip_rates = df.groupby("model")["flipped"].mean().reset_index().rename(columns={"flipped": "flip_rate"})
    transition_by_mode_strength = df.groupby(["model", "mode", "strength", "transition_type"]).size().reset_index(name="count")


    # 7. TEACHER VS STUDENT (Only applies if both exist in the split)
    pair_index_cols = ["qid", "bucket", "mode", "strength", "run_id"]
    
    teacher_student_acc_gap = pd.DataFrame()
    error_inheritance = pd.DataFrame()
    regressive_inheritance = pd.DataFrame()

    # Check if there are both 'teacher' and 'student' models within this dataframe
    # This is relevant for family-specific sheets where we expect one teacher and one student per family.
    # For the combined sheet, this comparison is more complex and might be skipped or aggregated differently.
    
    # Identify unique model types (e.g., "Gemma_teacher", "Gemma_student")
    model_types = df["model"].unique()
    
    # Check if we have exactly one teacher and one student for a specific family
    # This logic is primarily for the family-specific sheets.
    has_teacher = any("teacher" in mt.lower() for mt in model_types)
    has_student = any("student" in mt.lower() for mt in model_types)

    if has_teacher and has_student:
        # Create pairs for comparison
        pairs = df.pivot_table(
            index=pair_index_cols,
            columns="model",
            values=["first_correct", "after_correct", "sycophancy", "first_label", "after_label"],
            aggfunc="first"
        ).reset_index()

        # Flatten columns
        pairs.columns = ["_".join([str(c) for c in col if c not in [""]]) for col in pairs.columns]
        
        # Dynamically find the teacher and student columns for 'after_correct'
        teacher_after_col = next((c for c in pairs.columns if 'after_correct' in c and 'teacher' in c), None)
        student_after_col = next((c for c in pairs.columns if 'after_correct' in c and 'student' in c), None)

        if teacher_after_col and student_after_col:
            # Fill NaNs with 0 (False) to avoid float->int errors if data is missing
            pairs[teacher_after_col] = pairs[teacher_after_col].fillna(0).astype(int)
            pairs[student_after_col] = pairs[student_after_col].fillna(0).astype(int)
            
            pairs["acc_gap_after"] = pairs[teacher_after_col] - pairs[student_after_col]
            teacher_student_acc_gap = pairs.groupby(["mode", "strength"])["acc_gap_after"].mean().reset_index().rename(columns={"acc_gap_after": "mean_acc_gap_after"})

            # Error Inheritance
            pairs["teacher_wrong_after"] = ~pairs[teacher_after_col].astype(bool)
            pairs["student_wrong_after"] = ~pairs[student_after_col].astype(bool)
            error_inheritance = (
                pairs[pairs["teacher_wrong_after"]]
                .assign(inherited=lambda x: x["student_wrong_after"])
                .groupby(["mode", "strength"])["inherited"]
                .mean().reset_index()
                .rename(columns={"inherited": "error_inheritance_rate"})
            )
            
            # Regressive sycophancy inheritance
            teacher_syc_col = next((c for c in pairs.columns if 'sycophancy' in c and 'teacher' in c), None)
            student_syc_col = next((c for c in pairs.columns if 'sycophancy' in c and 'student' in c), None)

            if teacher_syc_col and student_syc_col:
                pairs["teacher_regressive"] = pairs[teacher_syc_col] == "regressive"
                pairs["student_regressive"] = pairs[student_syc_col] == "regressive"

                regressive_inheritance = (
                    pairs[pairs["teacher_regressive"]]
                    .assign(inherited=lambda x: x["student_regressive"])
                    .groupby(["mode", "strength"])["inherited"]
                    .mean()
                    .reset_index()
                    .rename(columns={"inherited": "regressive_inheritance_rate"})
                )


    # 8. CHAIN STABILITY
    # This section is commented out in the user's provided code, so keeping it minimal or skipped.
    # The original chain_stability_summary logic might need adjustment if 'model' is now 'Family_model_type'
    # For now, we'll return an empty DataFrame for chain_stability_summary as per the user's provided code.
    chain_stability_summary = pd.DataFrame()
    # chain_group_cols = ["qid", "model", "mode", "run_id"]
    # def chain_stable(g):
    #     first = g["after_label"].iloc[0]
    #     return int((g["after_label"] == first).all())
    # chain_stability = (
    #     df.groupby(chain_group_cols)
    #     .apply(chain_stable)
    #     .reset_index(name="stable_chain")
    # )
    # chain_stability_summary = (
    #     chain_stability.groupby(["model", "mode"])["stable_chain"]
    #     .mean()
    #     .reset_index()
    #     .rename(columns={"stable_chain": "chain_stability_rate"})
    # )


    # 9. BUCKET CONDITIONAL
    bucket_acc_after = (
       df.groupby(["model", "bucket"])["after_correct"]
       .mean().reset_index().rename(columns={"after_correct": "acc_after_by_bucket"})
    )

    # --------------------------
    # 11. EXPORT TO EXCEL (CONSOLIDATED)
    # --------------------------
    
    # Helper to round floats to 2 decimal places
    def round_df(d):
        if d is None or d.empty:
            return d
        # Round only float columns
        xml_df = d.copy()
        for col in xml_df.select_dtypes(include=['float']).columns:
            xml_df[col] = xml_df[col].round(2)
        return xml_df

    dfs_to_export = [
        ("Accuracy Summary", round_df(accuracy_summary)),
        ("Bucket Summary", round_df(bucket_summary)),
        ("Sycophancy Counts", round_df(sycophancy_summary)),
        ("Overall Sycophancy Rates", round_df(sycophancy_rates)),
        ("Sycophancy Rates by Mode", round_df(sycophancy_by_mode)),
        ("Sycophancy Rates by Strength", round_df(sycophancy_by_strength)),
        ("Accuracy: Mode & Strength", round_df(accuracy_by_mode_strength)),
        ("Pivot: Strength & Mode (First)", round_df(strength_mode_pivot)),
        ("Pivot: Strength & Mode (After)", round_df(strength_mode_pivot_after)),
        ("Transition Summary", round_df(transition_summary)),
        ("Flip Rates", round_df(flip_rates)),
        ("Transition by Mode/Strength", round_df(transition_by_mode_strength)),
        ("Teacher-Student Gap", round_df(teacher_student_acc_gap)),
        ("Error Inheritance", round_df(error_inheritance)),
        ("Regressive Inheritance", round_df(regressive_inheritance)), 
        ("Bucket Conditional Accuracy", round_df(bucket_acc_after))
    ]

    return dfs_to_export

def write_sheet(writer, sheet_name, tables):
    current_row = 0
    for title, df_out in tables:
        if df_out is None or df_out.empty: continue
        
        # Write title
        pd.DataFrame([title]).to_excel(writer, sheet_name=sheet_name, startrow=current_row, startcol=0, index=False, header=False)
        
        # Write table
        df_out.to_excel(writer, sheet_name=sheet_name, startrow=current_row + 1, startcol=0, index=False)
        
        # Update current_row for next table (+1 for title, +1 for header, +len(df) for rows, +2 for spacing)
        current_row += len(df_out) + 4


# --------------------------
# MAIN EXECUTION
# --------------------------
if __name__ == "__main__":
    
    all_records = []
    
    # 1. LOAD AND TAG
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}")
            continue
            
        print(f"Loading {file_path}...")
        
        # Determine Family
        fname = os.path.basename(file_path).lower()
        if "gemma" in fname: family = "Gemma"
        elif "llama" in fname: family = "Llama"
        elif "nvidia" in fname: family = "Nvidia"
        else: family = "Other"
        
        with open(file_path, "r") as f:
            data = json.load(f)
            recs = data["records"]
            
            # Tagging
            for r in recs:
                r["family"] = family
                original_model = r["model"] # 'teacher' or 'student'
                # Make model name unique globally: "Gemma_teacher"
                r["model"] = f"{family}_{original_model}"
            
            all_records.extend(recs)

    full_df = pd.DataFrame(all_records)
    
    # 2. GENERATE TABLES
    
    # A) Combined Analysis
    print("Analyzing Combined Data...")
    combined_tables = analyze_dataframe(full_df)
    
    # B) Family Analysis
    family_tables = {}
    families = full_df["family"].unique()
    
    for fam in families:
        print(f"Analyzing {fam} Data...")
        sub_df = full_df[full_df["family"] == fam]
        family_tables[fam] = analyze_dataframe(sub_df)
        
    # 3. WRITE TO EXCEL
    print(f"Writing to {output_file}...")
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # Write Combined First
        write_sheet(writer, "Combined_Results", combined_tables)
        
        # Write Families
        for fam in sorted(families):
            sheet_name = f"{fam}_Results"
            write_sheet(writer, sheet_name, family_tables[fam])
            
    print("Done!")