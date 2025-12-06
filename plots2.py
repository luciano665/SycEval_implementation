import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np

# Use seaborn style
sns.set_theme(style="whitegrid")

# WVU Colors
TEACHER_COLOR = '#EAAA00' # Gold
STUDENT_COLOR = '#002855' # Dark Blue

# --------------------------
# 1. DATA LOADING
# --------------------------
input_files = [
    "results/gemma_distill_results_1000.json",
    "results/llama3_2_distill_results_1000.json",
    "results/nvidia_distill_results_1000.json"
]

def load_data():
    all_records = []
    print("Loading data for plots...")
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Skipping {file_path}")
            continue
            
        # Determine Family
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
    
    # Pre-calculations
    df["first_correct"] = df["first_label"] == "correct"
    df["after_correct"] = df["after_label"] == "correct"
    
    df["is_sycophantic"] = df["sycophancy"] != "none"
    df["is_regressive"] = df["sycophancy"] == "regressive"
    df["is_progressive"] = df["sycophancy"] == "progressive"
    
    return df

# --------------------------
# 2. PLOTTING FUNCTIONS
# --------------------------

def plot_sycophancy_types(df):
    """
    Side-by-side comparison of Regressive and Progressive Sycophancy.
    Colored by Teacher/Student.
    Ordered: Teacher First.
    Labels: 2 decimals.
    """
    data = []
    for role, group in df.groupby("role"):
        if role == "Unknown": continue
        
        # Regressive Rate (from correct)
        n_correct = group["first_correct"].sum()
        reg_rate = group["is_regressive"].sum() / n_correct if n_correct > 0 else 0
        
        # Progressive Rate (from incorrect)
        n_incorrect = (~group["first_correct"]).sum()
        prog_rate = group["is_progressive"].sum() / n_incorrect if n_incorrect > 0 else 0
        
        data.append({"Role": role, "Type": "Regressive (Abandon Truth)", "Rate": reg_rate})
        # Corrected Label
        data.append({"Role": role, "Type": "Progressive (Corrected to Right Answer)", "Rate": prog_rate})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(
        data=plot_df,
        x="Type",
        y="Rate",
        hue="Role",
        hue_order=["Teacher", "Student"], # Teacher First
        palette={ "Teacher": TEACHER_COLOR, "Student": STUDENT_COLOR },
        edgecolor="black"
    )
    
    plt.title("Sycophancy Types: Regressive vs Progressive", fontsize=14, fontweight='bold')
    plt.ylabel("Sycophancy Rate", fontsize=11)
    plt.xlabel("")
    plt.ylim(0, 0.6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.savefig("Plot_Sycophancy_Types.png", dpi=300)
    plt.close()
    print("Saved Plot_Sycophancy_Types.png")


def plot_accuracy_impact(df):
    """
    Accuracy Before vs After on X-axis.
    Colored by Teacher/Student.
    Ordered: Teacher First.
    Labels: 2 decimals.
    """
    data = []
    for role, group in df.groupby("role"):
        if role == "Unknown": continue
        
        acc_before = group["first_correct"].mean()
        acc_after = group["after_correct"].mean()
        
        data.append({"Role": role, "Time": "Before Intervention", "Accuracy": acc_before})
        data.append({"Role": role, "Time": "After Intervention", "Accuracy": acc_after})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(
        data=plot_df,
        x="Time",
        y="Accuracy",
        hue="Role", 
        hue_order=["Teacher", "Student"], # Teacher First
        palette={ "Teacher": TEACHER_COLOR, "Student": STUDENT_COLOR },
        edgecolor="black"
    )
    
    plt.title("Impact of Sycophancy on Accuracy", fontsize=14, fontweight='bold')
    plt.ylabel("Average Accuracy", fontsize=11)
    plt.xlabel("")
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.savefig("Plot_Accuracy_Impact.png", dpi=300)
    plt.close()
    print("Saved Plot_Accuracy_Impact.png")


def plot_mode_comparison(df):
    """
    Preemptive vs In-context: Total Sycophancy Rate.
    Colored by Teacher/Student.
    Ordered: Teacher First.
    Labels: 2 decimals.
    """
    data = []
    for (role, mode), group in df.groupby(["role", "mode"]):
        if role == "Unknown": continue
        
        # Calculate Total Sycophancy Rate
        total_attempts = len(group)
        rate = group["is_sycophantic"].sum() / total_attempts if total_attempts > 0 else 0
        
        data.append({"Role": role, "Mode": mode.capitalize(), "Rate": rate})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(
        data=plot_df,
        x="Mode",
        y="Rate",
        hue="Role",
        hue_order=["Teacher", "Student"], # Teacher First
        palette={ "Teacher": TEACHER_COLOR, "Student": STUDENT_COLOR },
        edgecolor="black"
    )
    
    plt.title("Vulnerability by Pressure Mode (Total Sycophancy)", fontsize=14, fontweight='bold')
    plt.ylabel("Total Sycophancy Rate", fontsize=11)
    plt.xlabel("Pressure Type")
    plt.ylim(0, 0.6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.savefig("Plot_Mode_Comparison.png", dpi=300)
    plt.close()
    print("Saved Plot_Mode_Comparison.png")

def plot_rebuttal_comparison(df):
    """
    Rebuttal Type on X-axis: Total Sycophancy Rate.
    Colored by Teacher/Student.
    Ordered: Teacher First.
    Labels: 2 decimals.
    """
    data = []
    for (role, strength), group in df.groupby(["role", "strength"]):
        if role == "Unknown": continue
        
        # Calculate Total Sycophancy Rate
        total_attempts = len(group)
        rate = group["is_sycophantic"].sum() / total_attempts if total_attempts > 0 else 0
        
        data.append({"Role": role, "Strength": strength.capitalize(), "Rate": rate})
        
    plot_df = pd.DataFrame(data)
    
    # Custom Sort for Strengths
    strength_order = ["Simple", "Ethos", "Justification", "Citation"]
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        data=plot_df,
        x="Strength",
        y="Rate",
        hue="Role",
        hue_order=["Teacher", "Student"], # Teacher First
        order=strength_order, # Logical strength order
        palette={ "Teacher": TEACHER_COLOR, "Student": STUDENT_COLOR },
        edgecolor="black"
    )
    
    plt.title("Robustness by Rebuttal Type (Total Sycophancy)", fontsize=14, fontweight='bold')
    plt.ylabel("Total Sycophancy Rate", fontsize=11)
    plt.xlabel("Argument Type")
    plt.ylim(0, 0.6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type")
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.savefig("Plot_Rebuttal_Comparison.png", dpi=300)
    plt.close()
    print("Saved Plot_Rebuttal_Comparison.png")

def plot_overall_model_comparison(df):
    """
    New Graph: Total Sycophancy Rate by Model Family.
    X-axis: Model (Gemma, Llama, Nvidia).
    Y-axis: Total Sycophancy Rate.
    Colored by Teacher/Student.
    Ordered: Teacher First.
    """
    data = []
    for (family, role), group in df.groupby(["family", "role"]):
        if role == "Unknown": continue
        
        # Calculate Total Sycophancy Rate
        total_attempts = len(group)
        rate = group["is_sycophantic"].sum() / total_attempts if total_attempts > 0 else 0
        
        data.append({"Family": family, "Role": role, "Rate": rate})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    ax = sns.barplot(
        data=plot_df,
        x="Family",
        y="Rate",
        hue="Role",
        hue_order=["Teacher", "Student"], # Teacher First
        palette={ "Teacher": TEACHER_COLOR, "Student": STUDENT_COLOR },
        edgecolor="black"
    )
    
    plt.title("Overall Sycophancy by Model Family", fontsize=14, fontweight='bold')
    plt.ylabel("Total Sycophancy Rate", fontsize=11)
    plt.xlabel("Model Family")
    plt.ylim(0, 0.6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.tight_layout()
    plt.savefig("Plot_Overall_Model_Comparison.png", dpi=300)
    plt.close()
    print("Saved Plot_Overall_Model_Comparison.png")


if __name__ == "__main__":
    df = load_data()
    if not df.empty:
        plot_sycophancy_types(df)
        plot_accuracy_impact(df)
        plot_mode_comparison(df)
        plot_rebuttal_comparison(df)
        plot_overall_model_comparison(df)
    else:
        print("No data found.")