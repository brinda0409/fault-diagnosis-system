import pandas as pd
import numpy as np
import json
import os

csv_path = r"d:\FAULT DIAGNOSIS\Data CWRU\feature_time_48k_2048_load_1.csv"

# 1. Load CSV
df = pd.read_csv(csv_path)

print("=== 1. DATASET DIMENSIONS ===")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print()

print("=== 2. COLUMN NAMES & DATA TYPES ===")
print(df.dtypes)
print()

print("=== 3. FIRST 5 ROWS ===")
print(df.head())
print()

print("=== 4. LAST 5 ROWS ===")
print(df.tail())
print()

# Identify target column: usually the last column or named 'label' / 'target' / 'fault' / 'class' / 'fault_type'
# Let's inspect column names
target_col = df.columns[-1]
feature_cols = df.columns[:-1]

print(f"Identified Target Column: {target_col}")
print(f"Identified Feature Columns ({len(feature_cols)}): {list(feature_cols)}")
print()

print("=== 5. TARGET CLASS DISTRIBUTION ===")
val_counts = df[target_col].value_counts()
print("Unique target classes count:", len(val_counts))
print("Class names and sample counts:")
print(val_counts)
print("\nPercentage distribution:")
print(df[target_col].value_counts(normalize=True) * 100)
print()

print("=== 6. DATA QUALITY CHECKS ===")
# Missing values
missing_per_col = df.isnull().sum()
total_missing = missing_per_col.sum()
print(f"Total missing values (null/NaN): {total_missing}")
if total_missing > 0:
    print(missing_per_col[missing_per_col > 0])

# Duplicate rows
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Infinite values in numerical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
total_infs = inf_counts.sum()
print(f"Total infinite values (+inf / -inf): {total_infs}")
if total_infs > 0:
    print(inf_counts[inf_counts > 0])

# Non-numeric values in feature columns
non_numeric_features = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
print(f"Non-numeric feature columns: {non_numeric_features}")
print()

print("=== 7. DESCRIPTIVE STATISTICS FOR FEATURES ===")
desc_stats = df[feature_cols].describe().T
print(desc_stats.to_string())

# Save summary metrics to a json or txt file for precise report generation
summary = {
    "num_rows": int(df.shape[0]),
    "num_cols": int(df.shape[1]),
    "columns": list(df.columns),
    "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
    "target_col": target_col,
    "feature_cols": list(feature_cols),
    "class_counts": {str(k): int(v) for k, v in val_counts.items()},
    "missing_values": int(total_missing),
    "duplicate_rows": int(num_duplicates),
    "infinite_values": int(total_infs),
    "non_numeric_features": non_numeric_features,
    "head_5": df.head().to_dict(orient="records"),
    "tail_5": df.tail().to_dict(orient="records"),
    "describe": desc_stats.to_dict(orient="index")
}

with open(r"d:\FAULT DIAGNOSIS\inspection_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nInspection complete. Results saved to inspection_results.json")
