import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Paths configuration
prep_dir = r"d:\FAULT DIAGNOSIS\results\preprocessing"
output_dir = r"d:\FAULT DIAGNOSIS\results\models"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Preprocessed Datasets & Label Mapping
print("=== 1. LOADING PREPROCESSED DATASETS ===")

X_train_unscaled = pd.read_csv(os.path.join(prep_dir, 'X_train_unscaled.csv'))
X_test_unscaled = pd.read_csv(os.path.join(prep_dir, 'X_test_unscaled.csv'))

X_train_scaled = pd.read_csv(os.path.join(prep_dir, 'X_train_scaled.csv'))
X_test_scaled = pd.read_csv(os.path.join(prep_dir, 'X_test_scaled.csv'))

y_train = pd.read_csv(os.path.join(prep_dir, 'y_train.csv'))['fault_label'].values
y_test = pd.read_csv(os.path.join(prep_dir, 'y_test.csv'))['fault_label'].values

with open(os.path.join(prep_dir, 'label_mapping.json'), 'r') as f:
    label_mapping = json.load(f)

# Convert string keys in label_mapping to integers
label_mapping = {int(k): v for k, v in label_mapping.items()}
class_names = [label_mapping[i] for i in range(len(label_mapping))]

feature_names = list(X_train_unscaled.columns)

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"Class Names ({len(class_names)}): {class_names}")
print(f"X_train shape: {X_train_unscaled.shape}, X_test shape: {X_test_unscaled.shape}")

# ------------------------------------------------------------------------------
# 2. MODEL 1: RANDOM FOREST CLASSIFIER (UNSCALED FEATURES)
# ------------------------------------------------------------------------------
print("\n=== 2. TRAINING RANDOM FOREST CLASSIFIER (BASELINE) ===")
rf_model = RandomForestClassifier(random_state=42)

# Measure training time
t0 = time.perf_counter()
rf_model.fit(X_train_unscaled, y_train)
rf_train_time = time.perf_counter() - t0

# Measure test prediction time
t0 = time.perf_counter()
rf_preds = rf_model.predict(X_test_unscaled)
rf_pred_time = time.perf_counter() - t0

# Calculate evaluation metrics (weighted)
rf_acc = accuracy_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds, average='weighted')
rf_rec = recall_score(y_test, rf_preds, average='weighted')
rf_f1 = f1_score(y_test, rf_preds, average='weighted')
rf_report_dict = classification_report(y_test, rf_preds, target_names=class_names, output_dict=True)
rf_report_text = classification_report(y_test, rf_preds, target_names=class_names)
rf_cm = confusion_matrix(y_test, rf_preds)

print(f"Random Forest Training Time   : {rf_train_time:.4f} seconds")
print(f"Random Forest Prediction Time : {rf_pred_time:.4f} seconds")
print(f"Accuracy  : {rf_acc:.4f}")
print(f"Precision : {rf_prec:.4f}")
print(f"Recall    : {rf_rec:.4f}")
print(f"F1-Score  : {rf_f1:.4f}")

# Save RF results
rf_results = {
    "model_name": "Random Forest Classifier",
    "scaled_data_used": False,
    "hyperparameters": rf_model.get_params(),
    "training_time_seconds": rf_train_time,
    "prediction_time_seconds": rf_pred_time,
    "metrics": {
        "accuracy": float(rf_acc),
        "precision_weighted": float(rf_prec),
        "recall_weighted": float(rf_rec),
        "f1_score_weighted": float(rf_f1)
    },
    "classification_report": rf_report_dict
}
with open(os.path.join(output_dir, 'random_forest_results.json'), 'w') as f:
    json.dump(rf_results, f, indent=2)

joblib.dump(rf_model, os.path.join(output_dir, 'random_forest_baseline.joblib'))

# Save RF Confusion Matrix Plot
plt.figure(figsize=(10, 8))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('Random Forest - Confusion Matrix (Baseline)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Fault Class', fontsize=12)
plt.ylabel('Actual Fault Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'random_forest_confusion_matrix.png'))
plt.close()

# RF Feature Importances
rf_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

rf_importances.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
print("\nRandom Forest Feature Importances:")
print(rf_importances.to_string(index=False))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=rf_importances, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
plt.title('Random Forest - Feature Importance Ranking', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Gini Importance Weight', fontsize=12)
plt.ylabel('Vibration Feature', fontsize=12)
for idx, row in rf_importances.iterrows():
    plt.text(row['Importance'] + 0.005, idx, f"{row['Importance']:.4f}", va='center', fontsize=10)
plt.xlim(0, max(rf_importances['Importance']) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'random_forest_feature_importance.png'))
plt.close()

# ------------------------------------------------------------------------------
# 3. MODEL 2: SUPPORT VECTOR MACHINE (SCALED FEATURES)
# ------------------------------------------------------------------------------
print("\n=== 3. TRAINING SUPPORT VECTOR MACHINE (BASELINE) ===")
svm_model = SVC(kernel='rbf', random_state=42)

# Measure training time
t0 = time.perf_counter()
svm_model.fit(X_train_scaled, y_train)
svm_train_time = time.perf_counter() - t0

# Measure test prediction time
t0 = time.perf_counter()
svm_preds = svm_model.predict(X_test_scaled)
svm_pred_time = time.perf_counter() - t0

# Calculate evaluation metrics (weighted)
svm_acc = accuracy_score(y_test, svm_preds)
svm_prec = precision_score(y_test, svm_preds, average='weighted')
svm_rec = recall_score(y_test, svm_preds, average='weighted')
svm_f1 = f1_score(y_test, svm_preds, average='weighted')
svm_report_dict = classification_report(y_test, svm_preds, target_names=class_names, output_dict=True)
svm_report_text = classification_report(y_test, svm_preds, target_names=class_names)
svm_cm = confusion_matrix(y_test, svm_preds)

print(f"SVM Training Time   : {svm_train_time:.4f} seconds")
print(f"SVM Prediction Time : {svm_pred_time:.4f} seconds")
print(f"Accuracy  : {svm_acc:.4f}")
print(f"Precision : {svm_prec:.4f}")
print(f"Recall    : {svm_rec:.4f}")
print(f"F1-Score  : {svm_f1:.4f}")

# Save SVM results
svm_results = {
    "model_name": "Support Vector Machine (RBF Kernel)",
    "scaled_data_used": True,
    "hyperparameters": svm_model.get_params(),
    "training_time_seconds": svm_train_time,
    "prediction_time_seconds": svm_pred_time,
    "metrics": {
        "accuracy": float(svm_acc),
        "precision_weighted": float(svm_prec),
        "recall_weighted": float(svm_rec),
        "f1_score_weighted": float(svm_f1)
    },
    "classification_report": svm_report_dict
}
with open(os.path.join(output_dir, 'svm_results.json'), 'w') as f:
    json.dump(svm_results, f, indent=2)

joblib.dump(svm_model, os.path.join(output_dir, 'svm_baseline.joblib'))

# Save SVM Confusion Matrix Plot
plt.figure(figsize=(10, 8))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.title('SVM (RBF Kernel) - Confusion Matrix (Baseline)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Fault Class', fontsize=12)
plt.ylabel('Actual Fault Class', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'svm_confusion_matrix.png'))
plt.close()

# ------------------------------------------------------------------------------
# 4. MODEL COMPARISON
# ------------------------------------------------------------------------------
print("\n=== 4. MODEL COMPARISON ===")
comparison_df = pd.DataFrame([
    {
        'Model': 'Random Forest Classifier',
        'Accuracy': rf_acc,
        'Precision': rf_prec,
        'Recall': rf_rec,
        'F1-score': rf_f1,
        'Training Time (s)': rf_train_time,
        'Prediction Time (s)': rf_pred_time
    },
    {
        'Model': 'SVM (RBF Kernel)',
        'Accuracy': svm_acc,
        'Precision': svm_prec,
        'Recall': svm_rec,
        'F1-score': svm_f1,
        'Training Time (s)': svm_train_time,
        'Prediction Time (s)': svm_pred_time
    }
])

comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
print(comparison_df.to_string(index=False))

# Plot Comparison Bar Chart
metrics_melted = pd.melt(
    comparison_df, 
    id_vars=['Model'], 
    value_vars=['Accuracy', 'Precision', 'Recall', 'F1-score'],
    var_name='Metric', 
    value_name='Score'
)

plt.figure(figsize=(10, 6))
chart = sns.barplot(data=metrics_melted, x='Metric', y='Score', hue='Model', palette='Set2')
plt.title('Baseline Model Performance Comparison (Random Forest vs. SVM)', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
plt.ylim(0, 1.15)
for p in chart.patches:
    height = p.get_height()
    if height > 0:
        chart.annotate(f'{height:.4f}',
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
plt.close()

print(f"\nPhase 4 Baseline Model Execution Completed! All outputs saved to {output_dir}")
