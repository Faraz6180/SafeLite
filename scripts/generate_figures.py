#!/usr/bin/env python3
"""
Generate publication‑ready figures and tables from experiment results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_parquet("results/raw/experiment.parquet")

# Aggregate success rates by task and baseline
agg = df.groupby(['task', 'baseline'])['success'].agg(['mean', 'std', 'count']).reset_index()
agg['se'] = agg['std'] / np.sqrt(agg['count'])

# Pivot for plotting
pivot_mean = agg.pivot(index='task', columns='baseline', values='mean')
pivot_se = agg.pivot(index='task', columns='baseline', values='se')

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(pivot_mean.index))  # task positions
width = 0.35  # bar width

baselines = pivot_mean.columns
colors = ['#2ecc71', '#e74c3c']  # B0 green, B3 red

for i, baseline in enumerate(baselines):
    means = pivot_mean[baseline].values
    errors = pivot_se[baseline].values
    ax.bar(x - width/2 + i*width, means, width, yerr=errors, label=baseline,
           color=colors[i], capsize=5, error_kw={'ecolor': 'black'})

ax.set_xticks(x)
ax.set_xticklabels(pivot_mean.index)
ax.set_ylabel("Success Rate")
ax.set_xlabel("Task")
ax.set_title("SafeLite: Success Rate by Task and Baseline (N=30)")
ax.legend(title="Baseline")
ax.set_ylim(0, 0.1)  # zoom in to show tiny differences if any
plt.tight_layout()

# Save figures
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/success_rate_bar_chart.pdf", dpi=300)
plt.savefig("figures/success_rate_bar_chart.png", dpi=300)
print("Figure saved to figures/")

# LaTeX Table
latex_table = pivot_mean.round(3)
latex_table.columns = ['B0 (No Safety)', 'B3 (SafeLite)']
latex_table.index.name = 'Task'
print("\n=== LaTeX Table ===")
print(latex_table.to_latex())