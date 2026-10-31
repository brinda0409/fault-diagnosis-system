import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# 1. Load Dataset
csv_path = r"d:\FAULT DIAGNOSIS\Data CWRU\feature_time_48k_2048_load_1.csv"
output_dir = r"d:\FAULT DIAGNOSIS\results\eda"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)
feature_cols = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
target_col = 'fault'

print("=== 1. VERIFY DIMENSIONS & DISTRIBUTION ===")
print(f"Dataset Shape: {df.shape}")
print("Class counts:")
print(df[target_col].value_counts())

# Order of fault classes for consistent plotting
fault_order = [
    'Normal_1', 
    'Ball_007_1', 'Ball_014_1', 'Ball_021_1',
    'IR_007_1', 'IR_014_1', 'IR_021_1',
    'OR_007_6_1', 'OR_014_6_1', 'OR_021_6_1'
]

# 2. Plot: Class Distribution Bar Chart
plt.figure(figsize=(10, 5))
ax = sns.countplot(data=df, x=target_col, order=fault_order, palette='viridis')
plt.title('Fault Class Sample Distribution (CWRU Dataset)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Fault Class / Bearing Operational Condition', fontsize=11)
plt.ylabel('Number of Samples', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=10)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
plt.close()

# 3. Plot: Feature Histograms (Distributions)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Distribution of Numerical Time-Domain Features', fontsize=16, fontweight='bold', y=0.99)
for idx, col in enumerate(feature_cols):
    ax = axes[idx // 3, idx % 3]
    sns.histplot(df[col], kde=True, ax=ax, color='navy', bins=30)
    ax.set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_histograms.png'))
plt.close()

# 4. Plot: Feature Boxplots (Outliers Detection)
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Boxplots of Numerical Features (Global Outlier Check)', fontsize=16, fontweight='bold', y=0.99)
for idx, col in enumerate(feature_cols):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(y=df[col], ax=ax, color='skyblue', flierprops=dict(marker='o', markerfacecolor='r', markersize=4))
    ax.set_title(f'Feature: {col}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
plt.close()

# 5. Plot: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Vibration Time-Domain Features', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# 6. Plot: RMS by Fault Class
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=target_col, y='rms', order=fault_order, palette='Set2')
plt.title('Vibration Energy (RMS) across 10 Fault Classes', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Fault Class', fontsize=11)
plt.ylabel('RMS (Root Mean Square Vibration Amplitude)', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rms_by_fault_class.png'))
plt.close()

# 7. Plot: Kurtosis by Fault Class
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x=target_col, y='kurtosis', order=fault_order, palette='Set2')
plt.title('Vibration Impulsiveness (Kurtosis) across 10 Fault Classes', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Fault Class', fontsize=11)
plt.ylabel('Kurtosis (Signal Peakiness)', fontsize=11)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kurtosis_by_fault_class.png'))
plt.close()

# 8. Summary Statistics Per Fault Class
class_stats = df.groupby(target_col)[feature_cols].agg(['mean', 'std', 'min', 'max'])
class_stats.to_csv(os.path.join(output_dir, 'summary_stats_by_class.csv'))

# Mean RMS and Kurtosis per class for rapid inspection
rms_kurtosis_summary = df.groupby(target_col)[['rms', 'kurtosis', 'sd', 'max']].agg(['mean', 'std']).reindex(fault_order)
print("\n=== RMS & KURTOSIS SUMMARY BY FAULT CLASS ===")
print(rms_kurtosis_summary)

# Outlier analysis using IQR method for each feature
outlier_report = {}
for col in feature_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_report[col] = {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": float(round(len(outliers) / len(df) * 100, 2))
    }

print("\n=== OUTLIER ANALYSIS (IQR METHOD) ===")
for k, v in outlier_report.items():
    print(f"Feature {k:10s}: {v['outlier_count']} outliers ({v['outlier_percentage']}%)")

# Save full metrics to JSON for reference
eda_results = {
    "shape": list(df.shape),
    "class_counts": df[target_col].value_counts().to_dict(),
    "correlation_matrix": corr_matrix.to_dict(),
    "outliers_iqr": outlier_report,
    "class_means": df.groupby(target_col)[feature_cols].mean().to_dict()
}
with open(os.path.join(output_dir, 'eda_results.json'), 'w') as f:
    json.dump(eda_results, f, indent=2)

print(f"\nEDA processing complete! All plots and statistics saved to {output_dir}")
