#!/usr/bin/env python3

import json
import pandas as pd

# --------------------------
# CONFIG
# --------------------------
input_file = "llama3.2_distill_results.json"  # Path to your JSON file
output_file = "evaluation_analysis.xlsx"      # Output Excel file

# --------------------------
# 1. LOAD JSON
# --------------------------
with open(input_file, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data["records"])

# --------------------------
# 2. MARK CORRECT RESPONSES
# --------------------------
df["first_correct"] = df["first_label"] == "correct"
df["after_correct"] = df["after_label"] == "correct"

# --------------------------
# 3. CALCULATE ACCURACY
# --------------------------
accuracy_summary = df.groupby("model")[["first_correct", "after_correct"]].mean().reset_index()
accuracy_summary.rename(columns={
    "first_correct": "accuracy_first",
    "after_correct": "accuracy_after"
}, inplace=True)

# Acurracy by mode and strength
accuracy_by_mode_strength = (
    df.groupby(["model", "mode", "strength"])[["first_correct", "after_correct"]]
    .mean()
    .reset_index()
    .rename(columns = {
        "first_correct": "acc_first",
        "after_correct": "acc_after"
    })
)
# --------------------------
# 4. BUCKET SUMMARY
# --------------------------
bucket_summary = df.groupby(["model", "bucket"]).size().unstack(fill_value=0).reset_index()

# --------------------------
# 5. SYCOPHANCY SUMMARY
# --------------------------
sycophancy_summary = df.groupby(["model", "sycophancy"]).size().unstack(fill_value=0).reset_index()

# Overall sycophancy rate per model
df["is_sycophantic"] = df["sycophancy"] != "none"
sycophancy_rates = (
    df.groupby("model")["is_sycophantic"]
    .mean()
    .reset_index()
    .rename(columns = {
        "is_sycophantic": "sycophancy_rate"
    })
)

# Regressive and progressive separately if present
if "regressive" in labels:
    df["is_regressive_syc"] = df["sycophancy"] == "regressive"
    regressive_rates = (
        df.groupby("model")["is_regressive_syc"]
        .mean()
        .reset_index()
        .rename(columns={"is_regressive_syc": "regressive_syc_rate"})
    )
else:
    regressive_rates = pd.DataFrame(columns=["model", "regressive_syc_rate"])

if "progressive" in labels:
    df["is_progressive_syc"] = df["sycophancy"] == "progressive"
    progressive_rates = (
        df.groupby("model")["is_progressive_syc"]
        .mean()
        .reset_index()
        .rename(columns={"is_progressive_syc": "progressive_syc_rate"})
    )
else:
    progressive_rates = pd.DataFrame(columns=["model", "progressive_syc_rate"])

# --------------------------
# 6. STRENGTH & MODE PIVOT (FIRST AND AFTER)
# --------------------------
# Pivot table for first correct
strength_mode_pivot = df.pivot_table(
    index="model",
    columns=["strength", "mode"],
    values="first_correct",
    aggfunc="mean"
).reset_index()

# Pivot table for after correct
strength_mode_pivot_after = df.pivot_table(
    index="model",
    columns=["strength", "mode"],
    values="after_correct",
    aggfunc="mean"
).reset_index()

# --------------------------
# 7. TRANSITION / PROGRESSION / REGRESSION
# --------------------------
def transition_type(row):
    first = row["first_label"]
    after = row["after_label"]
    first_ok = (first == "correct")
    after_ok = (after == "correct")
    if first_ok and after_ok:
        return "stable_correct"
    if (not first_ok) and (not after_ok):
        return "stable_incorrect"
    if (not first_ok) and after_ok:
        return "progression"
    if first_ok and (not after_ok):
        return "regression"
    return "other"

df["transition_type"] = df.apply(transition_type, axis=1)
df["flipped"] = df["first_label"] != df["after_label"]

# transition count
transition_summary = (
    df.groupby(["model", "transition"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

flip_rates = (
    df.groupby("model")["flipped"]
    .mean()
    .reset_index()
    .rename(columns={"flipped": "flip_rate"})
)

# Transition by mode/strength (for plots later)
transition_by_mode_strength = (
    df.groupby(["model", "mode", "strength", "transition"])
    .size()
    .reset_index(name="count")
)


# --------------------------
# 7. EXPORT TO EXCEL
# --------------------------
with pd.ExcelWriter(output_file) as writer:
    accuracy_summary.to_excel(writer, sheet_name="accuracy_summary", index=False)
    bucket_summary.to_excel(writer, sheet_name="bucket_summary", index=False)
    sycophancy_summary.to_excel(writer, sheet_name="sycophancy_summary", index=False)
    strength_mode_pivot.to_excel(writer, sheet_name="strength_mode_pivot", index=False)

print(f"Analysis complete! Results saved to {output_file}")