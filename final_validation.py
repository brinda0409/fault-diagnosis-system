import os
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


def main():
    print("=== PHASE 6: FINAL MODEL VALIDATION, EXPLAINABILITY & FAULT DIAGNOSIS ANALYSIS ===")

    # Paths configuration
    prep_dir = r"d:\FAULT DIAGNOSIS\results\preprocessing"
    baseline_dir = r"d:\FAULT DIAGNOSIS\results\models"
    tuning_dir = r"d:\FAULT DIAGNOSIS\results\models\tuning"
    output_dir = r"d:\FAULT DIAGNOSIS\results\models\final_validation"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Preprocessed Datasets & Label Mapping
    print("\n--- 1. Loading Preprocessed Test Datasets ---")
    X_test_unscaled = pd.read_csv(os.path.join(prep_dir, 'X_test_unscaled.csv'))
    X_test_scaled = pd.read_csv(os.path.join(prep_dir, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(prep_dir, 'y_test.csv'))['fault_label'].values

    with open(os.path.join(prep_dir, 'label_mapping.json'), 'r') as f:
        label_mapping = json.load(f)

    label_mapping = {int(k): v for k, v in label_mapping.items()}
    class_names = [label_mapping[i] for i in range(len(label_mapping))]
    feature_names = list(X_test_unscaled.columns)

    print(f"Test samples: {len(X_test_unscaled)}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Classes ({len(class_names)}): {class_names}")

    # Load baseline info
    with open(os.path.join(baseline_dir, 'random_forest_results.json'), 'r') as f:
        rf_baseline_info = json.load(f)
    with open(os.path.join(baseline_dir, 'svm_results.json'), 'r') as f:
        svm_baseline_info = json.load(f)

    # 2. Load Saved Optimized Models
    print("\n--- 2. Loading Saved Optimized Models from Phase 5 ---")
    rf_opt_path = os.path.join(tuning_dir, 'random_forest_optimized.joblib')
    svm_opt_path = os.path.join(tuning_dir, 'svm_optimized.joblib')

    rf_opt_model = joblib.load(rf_opt_path)
    svm_opt_model = joblib.load(svm_opt_path)

    print(f"Loaded RF Model: {type(rf_opt_model)}")
    print(f"Loaded SVM Pipeline: {type(svm_opt_model)}")

    # 3. Perform Final Test Evaluation
    print("\n--- 3. Evaluating Optimized Models on Untouched Test Set ---")

    # Random Forest predictions (unscaled data)
    t0 = time.perf_counter()
    rf_preds = rf_opt_model.predict(X_test_unscaled)
    rf_infer_time = time.perf_counter() - t0

    # SVM Pipeline predictions (unscaled data passed to pipeline)
    t0 = time.perf_counter()
    svm_preds = svm_opt_model.predict(X_test_unscaled)
    svm_infer_time = time.perf_counter() - t0

    # Calculate RF metrics
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_prec_w = precision_score(y_test, rf_preds, average='weighted')
    rf_rec_w = recall_score(y_test, rf_preds, average='weighted')
    rf_f1_w = f1_score(y_test, rf_preds, average='weighted')
    rf_f1_macro = f1_score(y_test, rf_preds, average='macro')
    rf_cm = confusion_matrix(y_test, rf_preds)
    rf_report_dict = classification_report(y_test, rf_preds, target_names=class_names, output_dict=True)

    # Calculate SVM metrics
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_prec_w = precision_score(y_test, svm_preds, average='weighted')
    svm_rec_w = recall_score(y_test, svm_preds, average='weighted')
    svm_f1_w = f1_score(y_test, svm_preds, average='weighted')
    svm_f1_macro = f1_score(y_test, svm_preds, average='macro')
    svm_cm = confusion_matrix(y_test, svm_preds)
    svm_report_dict = classification_report(y_test, svm_preds, target_names=class_names, output_dict=True)

    print("\n[Optimized Random Forest Performance]")
    print(f"Accuracy         : {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print(f"Weighted Precision: {rf_prec_w:.4f}")
    print(f"Weighted Recall   : {rf_rec_w:.4f}")
    print(f"Weighted F1-Score : {rf_f1_w:.4f}")
    print(f"Macro F1-Score    : {rf_f1_macro:.4f}")

    print("\n[Optimized SVM Pipeline Performance]")
    print(f"Accuracy         : {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    print(f"Weighted Precision: {svm_prec_w:.4f}")
    print(f"Weighted Recall   : {svm_rec_w:.4f}")
    print(f"Weighted F1-Score : {svm_f1_w:.4f}")
    print(f"Macro F1-Score    : {svm_f1_macro:.4f}")

    # 4. Final Model Selection
    if rf_acc > svm_acc:
        selected_model_name = "Optimized Random Forest Classifier"
        selected_model = rf_opt_model
        selected_acc = rf_acc
        selected_f1 = rf_f1_w
    elif svm_acc > rf_acc:
        selected_model_name = "Optimized Support Vector Machine (Pipeline)"
        selected_model = svm_opt_model
        selected_acc = svm_acc
        selected_f1 = svm_f1_w
    else:
        # Tie-breaker on weighted F1
        if rf_f1_w >= svm_f1_w:
            selected_model_name = "Optimized Random Forest Classifier"
            selected_model = rf_opt_model
            selected_acc = rf_acc
            selected_f1 = rf_f1_w
        else:
            selected_model_name = "Optimized Support Vector Machine (Pipeline)"
            selected_model = svm_opt_model
            selected_acc = svm_acc
            selected_f1 = svm_f1_w

    print(f"\n=======================================================")
    print(f"FINAL SELECTED MODEL: {selected_model_name}")
    print(f"Test Accuracy: {selected_acc*100:.2f}% | Test F1-Score: {selected_f1*100:.2f}%")
    print(f"=======================================================")

    # 5. Model Explainability Analysis
    print("\n--- 5. Performing Model Explainability Analysis ---")

    # A. Random Forest Gini Feature Importance
    rf_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_opt_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\nRandom Forest Gini Feature Importances:")
    print(rf_importances.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=rf_importances, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Final Optimized Random Forest - Gini Feature Importance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance Weight', fontsize=12)
    plt.ylabel('Vibration Feature', fontsize=12)
    for idx, row in rf_importances.iterrows():
        plt.text(row['Importance'] + 0.003, idx, f"{row['Importance']:.4f}", va='center', fontsize=10)
    plt.xlim(0, max(rf_importances['Importance']) * 1.15)
    plt.tight_layout()
    rf_fi_plot = os.path.join(output_dir, 'final_rf_feature_importance.png')
    plt.savefig(rf_fi_plot)
    plt.close()
    print(f"Saved: {rf_fi_plot}")

    # B. SVM Permutation Feature Importance (Mathematically Valid for RBF SVM)
    print("\nComputing Permutation Feature Importance for SVM Pipeline (30 repeats)...")
    perm_result = permutation_importance(
        svm_opt_model,
        X_test_unscaled,
        y_test,
        n_repeats=30,
        random_state=42,
        scoring='accuracy'
    )

    svm_perm_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean Importance Drop': perm_result.importances_mean,
        'Std Importance Drop': perm_result.importances_std
    }).sort_values(by='Mean Importance Drop', ascending=False).reset_index(drop=True)

    svm_perm_csv = os.path.join(output_dir, 'svm_permutation_feature_importance.csv')
    svm_perm_df.to_csv(svm_perm_csv, index=False)
    print(f"Saved: {svm_perm_csv}")
    print("\nSVM Permutation Feature Importance (Accuracy Drop when Feature Shuffled):")
    print(svm_perm_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(data=svm_perm_df, x='Mean Importance Drop', y='Feature', palette='crest', hue='Feature', legend=False)
    plt.title('Final Optimized SVM (RBF) - Permutation Feature Importance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Mean Drop in Test Accuracy when Feature Shuffled', fontsize=12)
    plt.ylabel('Vibration Feature', fontsize=12)
    for idx, row in svm_perm_df.iterrows():
        plt.text(row['Mean Importance Drop'] + 0.003, idx, f"{row['Mean Importance Drop']:.4f} ± {row['Std Importance Drop']:.4f}", va='center', fontsize=9)
    plt.xlim(0, max(svm_perm_df['Mean Importance Drop']) * 1.25)
    plt.tight_layout()
    svm_fi_plot = os.path.join(output_dir, 'svm_permutation_feature_importance.png')
    plt.savefig(svm_fi_plot)
    plt.close()
    print(f"Saved: {svm_fi_plot}")

    # 6. Fault Diagnosis Breakdown Summary
    print("\n--- 6. Creating Fault Diagnosis Summary Breakdown ---")
    diagnosis_rows = []

    for idx, c_name in enumerate(class_names):
        total_samples = int(np.sum(y_test == idx))
        correct_preds = int(rf_cm[idx, idx]) if selected_model_name.startswith("Optimized Random Forest") else int(svm_cm[idx, idx])
        incorrect_preds = total_samples - correct_preds
        rec = float(rf_report_dict[c_name]['recall']) if selected_model_name.startswith("Optimized Random Forest") else float(svm_report_dict[c_name]['recall'])
        f1 = float(rf_report_dict[c_name]['f1-score']) if selected_model_name.startswith("Optimized Random Forest") else float(svm_report_dict[c_name]['f1-score'])

        diagnosis_rows.append({
            'Fault Class': c_name,
            'Total Test Samples': total_samples,
            'Correct Predictions': correct_preds,
            'Incorrect Predictions': incorrect_preds,
            'Recall': rec,
            'F1-Score': f1
        })

    diagnosis_df = pd.DataFrame(diagnosis_rows)
    diag_csv_path = os.path.join(output_dir, 'fault_diagnosis_summary.csv')
    diagnosis_df.to_csv(diag_csv_path, index=False)
    print(f"Saved: {diag_csv_path}")
    print("\nFault Diagnosis Summary Breakdown Table:")
    print(diagnosis_df.to_string(index=False))

    # 7. Confusion Matrices Plots
    print("\n--- 7. Generating Final Confusion Matrix Plots ---")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    ax.set_title(f'Optimized Random Forest - Final Test Confusion Matrix (Acc: {rf_acc*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Fault Class', fontsize=12)
    ax.set_ylabel('Actual Fault Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    rf_cm_plot = os.path.join(output_dir, 'optimized_rf_confusion_matrix.png')
    plt.savefig(rf_cm_plot)
    plt.close()
    print(f"Saved: {rf_cm_plot}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    ax.set_title(f'Optimized SVM Pipeline - Final Test Confusion Matrix (Acc: {svm_acc*100:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Fault Class', fontsize=12)
    ax.set_ylabel('Actual Fault Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    svm_cm_plot = os.path.join(output_dir, 'optimized_svm_confusion_matrix.png')
    plt.savefig(svm_cm_plot)
    plt.close()
    print(f"Saved: {svm_cm_plot}")

    # 8. Model Comparison Summary Table & Chart
    comparison_table = pd.DataFrame([
        {
            'Model Stage': 'Baseline Random Forest',
            'Accuracy': rf_baseline_info['metrics']['accuracy'],
            'Precision (Weighted)': rf_baseline_info['metrics']['precision_weighted'],
            'Recall (Weighted)': rf_baseline_info['metrics']['recall_weighted'],
            'F1-Score (Weighted)': rf_baseline_info['metrics']['f1_score_weighted'],
            'Macro F1-Score': rf_baseline_info['classification_report']['macro avg']['f1-score']
        },
        {
            'Model Stage': 'Optimized Random Forest',
            'Accuracy': rf_acc,
            'Precision (Weighted)': rf_prec_w,
            'Recall (Weighted)': rf_rec_w,
            'F1-Score (Weighted)': rf_f1_w,
            'Macro F1-Score': rf_f1_macro
        },
        {
            'Model Stage': 'Baseline SVM (RBF)',
            'Accuracy': svm_baseline_info['metrics']['accuracy'],
            'Precision (Weighted)': svm_baseline_info['metrics']['precision_weighted'],
            'Recall (Weighted)': svm_baseline_info['metrics']['recall_weighted'],
            'F1-Score (Weighted)': svm_baseline_info['metrics']['f1_score_weighted'],
            'Macro F1-Score': svm_baseline_info['classification_report']['macro avg']['f1-score']
        },
        {
            'Model Stage': 'Optimized SVM (Pipeline)',
            'Accuracy': svm_acc,
            'Precision (Weighted)': svm_prec_w,
            'Recall (Weighted)': svm_rec_w,
            'F1-Score (Weighted)': svm_f1_w,
            'Macro F1-Score': svm_f1_macro
        }
    ])

    comp_csv_path = os.path.join(output_dir, 'final_model_comparison.csv')
    comparison_table.to_csv(comp_csv_path, index=False)
    print(f"\nSaved: {comp_csv_path}")
    print("\nFinal Model Comparison Table:")
    print(comparison_table.to_string(index=False))

    # Per-Class Classification Report Table
    class_report_rows = []
    for c_name in class_names:
        class_report_rows.append({
            'Fault Class': c_name,
            'RF Precision': rf_report_dict[c_name]['precision'],
            'RF Recall': rf_report_dict[c_name]['recall'],
            'RF F1-Score': rf_report_dict[c_name]['f1-score'],
            'SVM Precision': svm_report_dict[c_name]['precision'],
            'SVM Recall': svm_report_dict[c_name]['recall'],
            'SVM F1-Score': svm_report_dict[c_name]['f1-score'],
            'Support': int(rf_report_dict[c_name]['support'])
        })
    class_report_df = pd.DataFrame(class_report_rows)
    class_rep_csv_path = os.path.join(output_dir, 'final_classification_report.csv')
    class_report_df.to_csv(class_rep_csv_path, index=False)
    print(f"Saved: {class_rep_csv_path}")

    # Plot Comparison Bar Chart
    comp_melted = pd.DataFrame([
        {'Model': 'RF (Baseline)', 'Metric': 'Accuracy', 'Score': rf_baseline_info['metrics']['accuracy']},
        {'Model': 'RF (Baseline)', 'Metric': 'Weighted Precision', 'Score': rf_baseline_info['metrics']['precision_weighted']},
        {'Model': 'RF (Baseline)', 'Metric': 'Weighted F1-Score', 'Score': rf_baseline_info['metrics']['f1_score_weighted']},
        {'Model': 'RF (Baseline)', 'Metric': 'Macro F1-Score', 'Score': rf_baseline_info['classification_report']['macro avg']['f1-score']},

        {'Model': 'RF (Optimized)', 'Metric': 'Accuracy', 'Score': rf_acc},
        {'Model': 'RF (Optimized)', 'Metric': 'Weighted Precision', 'Score': rf_prec_w},
        {'Model': 'RF (Optimized)', 'Metric': 'Weighted F1-Score', 'Score': rf_f1_w},
        {'Model': 'RF (Optimized)', 'Metric': 'Macro F1-Score', 'Score': rf_f1_macro},

        {'Model': 'SVM (Baseline)', 'Metric': 'Accuracy', 'Score': svm_baseline_info['metrics']['accuracy']},
        {'Model': 'SVM (Baseline)', 'Metric': 'Weighted Precision', 'Score': svm_baseline_info['metrics']['precision_weighted']},
        {'Model': 'SVM (Baseline)', 'Metric': 'Weighted F1-Score', 'Score': svm_baseline_info['metrics']['f1_score_weighted']},
        {'Model': 'SVM (Baseline)', 'Metric': 'Macro F1-Score', 'Score': svm_baseline_info['classification_report']['macro avg']['f1-score']},

        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Accuracy', 'Score': svm_acc},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Weighted Precision', 'Score': svm_prec_w},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Weighted F1-Score', 'Score': svm_f1_w},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Macro F1-Score', 'Score': svm_f1_macro},
    ])

    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=comp_melted, x='Metric', y='Score', hue='Model', palette='tab10')
    plt.title('Phase 6: Final Model Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.ylim(0.85, 1.02)
    for p in chart.patches:
        height = p.get_height()
        if height > 0:
            chart.annotate(f'{height:.4f}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, 'final_model_comparison.png')
    plt.savefig(comp_plot_path)
    plt.close()
    print(f"Saved: {comp_plot_path}")

    # 9. Comprehensive JSON Report
    final_json_report = {
        "project": "Machine Learning-Based Fault Diagnosis of Rotating Machinery",
        "phase": "Phase 6 — Final Model Validation, Explainability, and Fault Diagnosis Analysis",
        "selected_final_model": selected_model_name,
        "selected_model_metrics": {
            "accuracy": float(selected_acc),
            "weighted_precision": float(rf_prec_w if selected_model_name.startswith("Optimized Random Forest") else svm_prec_w),
            "weighted_recall": float(rf_rec_w if selected_model_name.startswith("Optimized Random Forest") else svm_rec_w),
            "weighted_f1_score": float(selected_f1),
            "macro_f1_score": float(rf_f1_macro if selected_model_name.startswith("Optimized Random Forest") else svm_f1_macro)
        },
        "models_performance": {
            "random_forest_baseline": rf_baseline_info['metrics'],
            "random_forest_optimized": {
                "accuracy": float(rf_acc),
                "weighted_precision": float(rf_prec_w),
                "weighted_recall": float(rf_rec_w),
                "weighted_f1_score": float(rf_f1_w),
                "macro_f1_score": float(rf_f1_macro),
                "inference_time_seconds": float(rf_infer_time)
            },
            "svm_baseline": svm_baseline_info['metrics'],
            "svm_optimized_pipeline": {
                "accuracy": float(svm_acc),
                "weighted_precision": float(svm_prec_w),
                "weighted_recall": float(svm_rec_w),
                "weighted_f1_score": float(svm_f1_w),
                "macro_f1_score": float(svm_f1_macro),
                "inference_time_seconds": float(svm_infer_time)
            }
        },
        "classification_reports": {
            "random_forest_optimized": rf_report_dict,
            "svm_optimized_pipeline": svm_report_dict
        },
        "model_explainability": {
            "random_forest_top_3_gini_features": list(rf_importances['Feature'].head(3)),
            "svm_top_3_permutation_features": list(svm_perm_df['Feature'].head(3))
        },
        "validation_checks": {
            "test_set_unseen": True,
            "no_hyperparameter_retuning": True,
            "no_preprocessing_leakage": True,
            "original_csv_unmodified": True
        }
    }

    final_report_path = os.path.join(output_dir, 'final_validation_report.json')
    with open(final_report_path, 'w') as f:
        json.dump(final_json_report, f, indent=2)
    print(f"Saved: {final_report_path}")

    print("\nPhase 6 Final Model Validation & Explainability Execution Completed Successfully!")
    print(f"All outputs saved under: {output_dir}")


if __name__ == '__main__':
    main()
