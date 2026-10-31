import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Paths configuration
csv_path = r"d:\FAULT DIAGNOSIS\Data CWRU\feature_time_48k_2048_load_1.csv"
output_dir = r"d:\FAULT DIAGNOSIS\results\preprocessing"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load the original dataset
print("=== STEP 1: LOADING DATASET ===")
df = pd.read_csv(csv_path)
print(f"Loaded dataset shape: {df.shape}")

# Step 2: Separate features (X) and target (y)
feature_cols = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
target_col = 'fault'

X = df[feature_cols].copy()
y = df[target_col].copy()

# Step 3: Verify dataset dimensions & balance
print("\n=== STEP 3: VERIFYING DIMENSIONS & CLASS DISTRIBUTION ===")
print(f"Total samples: {len(df)}")
print(f"Number of input features: {X.shape[1]} -> {list(X.columns)}")
print(f"Number of target classes: {y.nunique()}")
print("Class counts in full dataset:")
print(y.value_counts())

# Step 4: Encode target classes with LabelEncoder
print("\n=== STEP 4: ENCODING CATEGORICAL TARGET LABELS ===")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Preserve label to class name mapping
label_mapping = {int(index): str(class_name) for index, class_name in enumerate(label_encoder.classes_)}
print("Label to Class Name Mapping:")
for k, v in label_mapping.items():
    print(f"  Encoded {k} -> '{v}'")

# Step 5: Perform Stratified Train/Test Split (80% Train, 20% Test)
print("\n=== STEP 5: STRATIFIED TRAIN/TEST SPLIT ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y_encoded, 
    test_size=0.20, 
    random_state=42, 
    stratify=y_encoded
)

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test shape  : {y_test.shape}")

# Verify stratified class distribution in train and test sets
train_dist = pd.Series(y_train).value_counts().sort_index()
test_dist = pd.Series(y_test).value_counts().sort_index()

class_dist_df = pd.DataFrame({
    'Encoded Label': list(label_mapping.keys()),
    'Fault Name': list(label_mapping.values()),
    'Train Count (80%)': train_dist.values,
    'Test Count (20%)': test_dist.values
})
print("\nTrain/Test Class Distribution:")
print(class_dist_df.to_string(index=False))

# Step 7: Feature Scaling with StandardScaler (Fitted ONLY on X_train to prevent leakage)
print("\n=== STEP 7: FEATURE SCALING (FITTING ON X_TRAIN ONLY) ===")
scaler = StandardScaler()

# Fit scaler on training data ONLY
scaler.fit(X_train)

# Transform both train and test feature sets
X_train_scaled_array = scaler.transform(X_train)
X_test_scaled_array = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames with feature column names
X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_cols, index=X_test.index)

print("Scaler mean per feature (calculated from X_train only):")
print(pd.Series(scaler.mean_, index=feature_cols))

# Step 8: Save artifacts to results/preprocessing/
print("\n=== STEP 8: SAVING PREPROCESSING ARTIFACTS ===")

# Save model transformers
joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))

with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
    json.dump(label_mapping, f, indent=2)

# Save unscaled datasets (for Random Forest)
X_train.to_csv(os.path.join(output_dir, 'X_train_unscaled.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test_unscaled.csv'), index=False)

# Save scaled datasets (for SVM)
X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)

# Save target vectors
pd.DataFrame({'fault_label': y_train}).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
pd.DataFrame({'fault_label': y_test}).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# Save comprehensive report summary
summary_report = {
    "original_dataset_size": len(df),
    "feature_list": feature_cols,
    "target_column": target_col,
    "encoding_method": "LabelEncoder",
    "label_mapping": label_mapping,
    "split_ratio": "80% Train / 20% Test",
    "random_state": 42,
    "stratify": True,
    "X_train_shape": list(X_train.shape),
    "X_test_shape": list(X_test.shape),
    "y_train_shape": list(y_train.shape),
    "y_test_shape": list(y_test.shape),
    "train_samples_per_class": 184,
    "test_samples_per_class": 46,
    "scaling_method": "StandardScaler",
    "scaler_fit_target": "X_train ONLY (No Data Leakage)",
    "data_leakage_prevented": True,
    "outliers_removed": False,
    "features_removed": False
}

with open(os.path.join(output_dir, 'preprocessing_report.json'), 'w') as f:
    json.dump(summary_report, f, indent=2)

print(f"Data Preprocessing Complete! All artifacts saved to {output_dir}")
