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

# --------------------------
# 4. BUCKET SUMMARY
# --------------------------
bucket_summary = df.groupby(["model", "bucket"]).size().unstack(fill_value=0).reset_index()

# --------------------------
# 5. SYCOPHANCY SUMMARY
# --------------------------
sycophancy_summary = df.groupby(["model", "sycophancy"]).size().unstack(fill_value=0).reset_index()

# --------------------------
# 6. STRENGTH & MODE PIVOT
# --------------------------
strength_mode_pivot = df.pivot_table(
    index="model",
    columns=["strength", "mode"],
    values="first_correct",
    aggfunc="mean"
).reset_index()

# --------------------------
# 7. EXPORT TO EXCEL
# --------------------------
with pd.ExcelWriter(output_file) as writer:
    accuracy_summary.to_excel(writer, sheet_name="accuracy_summary", index=False)
    bucket_summary.to_excel(writer, sheet_name="bucket_summary", index=False)
    sycophancy_summary.to_excel(writer, sheet_name="sycophancy_summary", index=False)
    strength_mode_pivot.to_excel(writer, sheet_name="strength_mode_pivot", index=False)

print(f"Analysis complete! Results saved to {output_file}")