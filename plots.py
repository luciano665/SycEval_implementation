import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. CONSOLIDATED DATA FOR ALL VISUALS ---

df_acc = pd.DataFrame({
    'model_family': ['Gemma', 'Llama', 'Nvidia'],
    'Acc_First_T': [0.890, 0.775, 0.628],
    'Acc_After_T': [0.507, 0.659, 0.075],
    'Acc_First_S': [0.731, 0.789, 0.345],
    'Acc_After_S': [0.435, 0.554, 0.131]
})
df_syc = pd.DataFrame({
    'model_family': ['Gemma', 'Llama', 'Nvidia'],
    'Syc_Student': [0.413, 0.335, 0.170],
    'Syc_Teacher': [0.455, 0.303, 0.083]
})
df_dyn_t = pd.DataFrame({
    'model_family': ['Gemma', 'Llama', 'Nvidia'],
    'Syc_Regressive': [0.411, 0.228, 0.063],
    'Syc_Progressive': [0.045, 0.075, 0.020]
})
df_dyn_s = pd.DataFrame({
    'model_family': ['Gemma', 'Llama', 'Nvidia'],
    'Syc_Regressive': [0.289, 0.258, 0.103],
    'Syc_Progressive': [0.123, 0.077, 0.067]
})
df_final_summary = pd.DataFrame({
    'model_family': ['Gemma', 'Gemma', 'Llama', 'Llama', 'Nvidia', 'Nvidia'],
    'model': ['student', 'teacher', 'student', 'teacher', 'student', 'teacher'],
    'Acc_First': [0.731, 0.890, 0.789, 0.775, 0.345, 0.628],
    'Acc_After': [0.435, 0.507, 0.554, 0.659, 0.131, 0.075],
    'Syc_Overall': [0.412, 0.455, 0.335, 0.303, 0.170, 0.083],
    'Syc_Regressive': [0.289, 0.411, 0.258, 0.228, 0.103, 0.063],
    'Syc_Progressive': [0.123, 0.045, 0.077, 0.075, 0.067, 0.020]
})
df_final_summary.sort_values(by=['model_family', 'model'], inplace=True)
df_final_summary = df_final_summary[['model_family', 'model', 'Acc_First', 'Acc_After', 
                                     'Syc_Overall', 'Syc_Regressive', 'Syc_Progressive']]


# --- COLOR DEFINITIONS ---
GOLD = '#FFC300' 
BLUE = '#1450B2'
EDGE_COLOR = '#333333'
EDGE_WIDTH = 0.8


# --- HELPER FUNCTION FOR LABELS ---
def autolabel(rects, ax, fontsize=9):
    """Add value labels on top of bars"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize, fontweight='bold')


# --- 2. GRAPH GENERATION FUNCTIONS ---

def generate_graph_1(df):
    """Graph 1: Teacher Model Accuracy"""
    labels = df['model_family']
    acc_first = df['Acc_First_T']
    acc_after = df['Acc_After_T']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, acc_first, width, 
                    label='Teacher Initial Accuracy (Before)', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, acc_after, width, 
                    label='Teacher Post-Intervention Accuracy (After)', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Accuracy Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Graph 1: Teacher Model Accuracy - Before vs. After Intervention', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    plt.tight_layout()
    plt.savefig('Graph_1_Teacher_Accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_graph_2(df):
    """Graph 2: Student Model Accuracy"""
    labels = df['model_family']
    acc_first = df['Acc_First_S']
    acc_after = df['Acc_After_S']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, acc_first, width, 
                    label='Student Initial Accuracy (Before)', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, acc_after, width, 
                    label='Student Post-Intervention Accuracy (After)', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Accuracy Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Graph 2: Student Model Accuracy - Before vs. After Intervention', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    
    plt.tight_layout()
    plt.savefig('Graph_2_Student_Accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_graph_3(df):
    """Graph 3: Teacher vs. Student Accuracy Gap"""
    df['gap_first'] = df['Acc_First_T'] - df['Acc_First_S']
    df['gap_after'] = df['Acc_After_T'] - df['Acc_After_S']
    
    labels = df['model_family']
    gap_first = df['gap_first']
    gap_after = df['gap_after']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, gap_first, width, 
                    label='Initial Gap (Teacher - Student)', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, gap_after, width, 
                    label='Post-Intervention Gap (Teacher - Student)', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Accuracy Difference (Acc_Teacher - Acc_Student)', 
                  fontsize=11, fontweight='bold')
    ax.set_title('Graph 3: Teacher vs. Student Accuracy Gap', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

    min_val = min(gap_first.min(), gap_after.min())
    max_val = max(gap_first.max(), gap_after.max())
    ax.set_ylim(min_val - 0.05, max_val + 0.05)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Graph_3_Accuracy_Gap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_graph_4(df):
    """Graph 4: Overall Sycophancy Rate"""
    labels = df['model_family']
    rate_student = df['Syc_Student']
    rate_teacher = df['Syc_Teacher']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, rate_student, width, 
                    label='Student Sycophancy Rate', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, rate_teacher, width, 
                    label='Teacher Sycophancy Rate', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Sycophancy Rate (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Graph 4: Overall Sycophancy Rate Comparison', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, max(rate_student.max(), rate_teacher.max()) * 1.15)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Graph_4_Sycophancy_Rates.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_graph_5(df):
    """Graph 5: Teacher Sycophancy Dynamics"""
    labels = df['model_family']
    rate_regressive = df['Syc_Regressive']
    rate_progressive = df['Syc_Progressive']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, rate_regressive, width, 
                    label='Teacher Regressive Sycophancy (Historical Bias)', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, rate_progressive, width, 
                    label='Teacher Progressive Sycophancy (Immediate Context Bias)', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Sycophancy Rate (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Graph 5: Teacher Model Sycophancy Dynamics', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, max(rate_regressive.max(), rate_progressive.max()) * 1.15)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Graph_5_Teacher_Sycophancy_Dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_graph_6(df):
    """Graph 6: Student Sycophancy Dynamics"""
    labels = df['model_family']
    rate_regressive = df['Syc_Regressive']
    rate_progressive = df['Syc_Progressive']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width/2, rate_regressive, width, 
                    label='Student Regressive Sycophancy (Historical Bias)', 
                    color=GOLD, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    rects2 = ax.bar(x + width/2, rate_progressive, width, 
                    label='Student Progressive Sycophancy (Immediate Context Bias)', 
                    color=BLUE, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    ax.set_ylabel('Sycophancy Rate (0.0 to 1.0)', fontsize=11, fontweight='bold')
    ax.set_title('Graph 6: Student Model Sycophancy Dynamics', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylim(0, max(rate_regressive.max(), rate_progressive.max()) * 1.15)
    
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Graph_6_Student_Sycophancy_Dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_table_5(df):
    """Table 5: Comprehensive Metric Summary"""
    df.to_csv('Table_5_Comprehensive_Summary.csv', index=False, float_format='%.3f')


# --- 3. MAIN EXECUTION ---
if __name__ == '__main__':
    # Generate Graphs in new order
    generate_graph_1(df_acc.copy())  # Teacher Accuracy
    generate_graph_2(df_acc.copy())  # Student Accuracy (moved up from 6)
    generate_graph_3(df_acc.copy())  # Accuracy Gap
    generate_graph_4(df_syc.copy())  # Overall Sycophancy
    generate_graph_5(df_dyn_t.copy()) # Teacher Dynamics
    generate_graph_6(df_dyn_s.copy()) # Student Dynamics
    
    # Save Table
    generate_table_5(df_final_summary.copy())