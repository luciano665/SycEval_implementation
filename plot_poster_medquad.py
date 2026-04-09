import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- DATA (MedQuad, N=1000) ---
# Format: (baseline_overall, conformal_overall, baseline_regressive, conformal_regressive)
data = {
    'Gemma\n(Small 1B)':  (0.363, 0.336, 0.255, 0.243),
    'Gemma\n(Large 4B)':  (0.490, 0.429, 0.400, 0.184),
    'LLaMA\n(Small 1B)':  (0.420, 0.404, 0.352, 0.321),
    'LLaMA\n(Large 3B)':  (0.436, 0.499, 0.346, 0.456),
    'Phi\n(Small 1.5B)':  (0.178, 0.163, 0.086, 0.074),
    'Phi\n(Large 2.7B)':  (0.251, 0.209, 0.091, 0.078),
}

labels        = list(data.keys())
baseline_ovr  = [v[0] for v in data.values()]
conformal_ovr = [v[1] for v in data.values()]
baseline_reg  = [v[2] for v in data.values()]
conformal_reg = [v[3] for v in data.values()]

# --- COLORS ---
C_BASE_OVR  = '#4878CF'
C_CONF_OVR  = '#6ACC65'
C_BASE_REG  = '#D65F5F'
C_CONF_REG  = '#F0A500'
EDGE_COLOR  = '#222222'
EDGE_WIDTH  = 0.7

# --- LAYOUT ---
n_groups = len(labels)
x = np.arange(n_groups)
width = 0.18
offsets = [-1.5, -0.5, 0.5, 1.5]

fig, ax = plt.subplots(figsize=(14, 7))

for i, (vals, color) in enumerate(zip(
    [baseline_ovr, conformal_ovr, baseline_reg, conformal_reg],
    [C_BASE_OVR, C_CONF_OVR, C_BASE_REG, C_CONF_REG]
)):
    ax.bar(x + offsets[i] * width, vals, width,
           color=color, edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, zorder=3)

# --- DELTA ANNOTATIONS ---
for i, (b_ovr, c_ovr, b_reg, c_reg) in enumerate(
    zip(baseline_ovr, conformal_ovr, baseline_reg, conformal_reg)
):
    for b_val, c_val, offset in [
        (b_ovr, c_ovr, offsets[1]),
        (b_reg, c_reg, offsets[3]),
    ]:
        delta = c_val - b_val
        sign  = '+' if delta >= 0 else ''
        color = '#B22222' if delta > 0.01 else ('#2E7D32' if delta < -0.01 else '#555555')
        ax.text(x[i] + offset * width, c_val + 0.012,
                f'{sign}{delta*100:.1f}%',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold', color=color)

# --- GROUP SEPARATORS ---
for xi in x[:-1]:
    ax.axvline(xi + 0.5, color='#cccccc', linewidth=1, linestyle='--', zorder=1)

# --- AXES ---
ax.set_ylabel('Sycophancy Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_title('Effect of Conformal Prediction on Sycophancy\nMedQuad Dataset (N=1000)',
             fontsize=14, fontweight='bold', pad=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax.set_axisbelow(True)

# --- LEGEND ---
legend_handles = [
    mpatches.Patch(color=C_BASE_OVR, label='Baseline — Overall'),
    mpatches.Patch(color=C_CONF_OVR, label='Conformal — Overall'),
    mpatches.Patch(color=C_BASE_REG, label='Baseline — Regressive'),
    mpatches.Patch(color=C_CONF_REG, label='Conformal — Regressive'),
]
ax.legend(handles=legend_handles, fontsize=10, framealpha=0.95,
          loc='upper right', ncol=2)


plt.tight_layout()
plt.savefig('poster_chart_medquad.png', dpi=300, bbox_inches='tight')
print("Saved: poster_chart_medquad.png")
plt.show()
