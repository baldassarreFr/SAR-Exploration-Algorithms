# scripts/aggregate_frontier.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

infile = "results/logs/frontier_big.csv"
outdir = Path("results/figures")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(infile)

# Basic checks
print("Rows:", len(df))
print(df.head())

# Convert types if necessary
df['time_to_find'] = pd.to_numeric(df['time_to_find'], errors='coerce')
df['coverage_percent'] = pd.to_numeric(df['coverage_percent'], errors='coerce')
df['success'] = pd.to_numeric(df['success'], errors='coerce')

# Summary statistics grouped by nothing (single method)
summary = {
    'n_episodes': len(df),
    'success_rate': df['success'].mean(),
    'time_median': df['time_to_find'].median(),
    'time_IQR': (df['time_to_find'].quantile(0.75) - df['time_to_find'].quantile(0.25)),
    'coverage_median': df['coverage_percent'].median(),
    'coverage_IQR': (df['coverage_percent'].quantile(0.75) - df['coverage_percent'].quantile(0.25)),
    'path_length_median': df['path_length'].median()
}
print("Summary:", summary)

# Save summary
pd.DataFrame([summary]).to_csv("results/logs/frontier_summary.csv", index=False)

# Boxplot time_to_find
plt.figure(figsize=(6,4))
plt.boxplot(df['time_to_find'].dropna(), vert=True)
plt.ylabel("time_to_find (timesteps)")
plt.title("Frontier - time_to_find")
plt.savefig(outdir / "frontier_time_boxplot.png", bbox_inches='tight')
plt.close()

# Histogram coverage_percent
plt.figure(figsize=(6,4))
plt.hist(df['coverage_percent'].dropna(), bins=30)
plt.xlabel("coverage_percent")
plt.ylabel("count")
plt.title("Frontier - coverage distribution")
plt.savefig(outdir / "frontier_coverage_hist.png", bbox_inches='tight')
plt.close()

# Save aggregated CSV (per-map stats)
per_map = df.groupby('map_id').agg({
    'time_to_find': ['median','count','mean'],
    'coverage_percent': ['median','mean'],
    'success': ['mean']
})
per_map.columns = ['_'.join(c) for c in per_map.columns]
per_map.to_csv("results/logs/frontier_per_map_stats.csv")
print("Saved summaries to results/logs/")