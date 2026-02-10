import matplotlib.pyplot as plt
import numpy as np

# Data provided by User
# Labels: Overall, Regressive, Progressive, In-Context, Preemptive
labels = ['Overall', 'Regressive\n(Correct->Wrong)', 'Progressive\n(Wrong->Right)', 'In-Context\n(No Defense)', 'Preemptive\n(Defense)']

baseline_data = [36.9, 28.9, 8.1, 35.6, 38.3]
conformal_data = [28.4, 20.8, 7.6, 21.6, 35.2]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, baseline_data, width, label='Llama Baseline (N=8320)', color='#88CCEE')
rects2 = ax.bar(x + width/2, conformal_data, width, label='Llama Conformal (N=6400)', color='#CC6677')

ax.set_ylabel('Percentage (%)')
ax.set_title('Llama Sycophancy: Baseline vs Conformal Evaluation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.1f%%')
ax.bar_label(rects2, padding=3, fmt='%.1f%%')

plt.tight_layout()
plt.savefig('results/final_run_v1/llama_breakdown_plot.png')
print("Plot saved to results/final_run_v1/llama_breakdown_plot.png")
