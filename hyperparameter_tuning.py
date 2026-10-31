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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300


def main():
    # Paths configuration
    prep_dir = r"d:\FAULT DIAGNOSIS\results\preprocessing"
    baseline_dir = r"d:\FAULT DIAGNOSIS\results\models"
    output_dir = r"d:\FAULT DIAGNOSIS\results\models\tuning"
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

    # Load baseline results for comparison
    with open(os.path.join(baseline_dir, 'random_forest_results.json'), 'r') as f:
        rf_baseline_info = json.load(f)

    with open(os.path.join(baseline_dir, 'svm_results.json'), 'r') as f:
        svm_baseline_info = json.load(f)

    rf_baseline_cm = confusion_matrix(y_test, joblib.load(os.path.join(baseline_dir, 'random_forest_baseline.joblib')).predict(X_test_unscaled))
    svm_baseline_cm = confusion_matrix(y_test, joblib.load(os.path.join(baseline_dir, 'svm_baseline.joblib')).predict(X_test_scaled))


    # ------------------------------------------------------------------------------
    # 2. RANDOM FOREST HYPERPARAMETER TUNING (UNSCALED FEATURES)
    # ------------------------------------------------------------------------------
    print("\n=== 2. RANDOM FOREST HYPERPARAMETER TUNING (GRIDSEARCH CV) ===")

    rf_param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    print(f"Random Forest Hyperparameter Grid: {rf_param_grid}")

    rf_base = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=rf_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        verbose=1
    )

    t0 = time.perf_counter()
    rf_grid.fit(X_train_unscaled, y_train)
    rf_tuning_time = time.perf_counter() - t0

    best_rf_model = rf_grid.best_estimator_
    best_rf_params = rf_grid.best_params_
    best_rf_cv_score = float(rf_grid.best_score_)

    print(f"\nRandom Forest Tuning Completed in {rf_tuning_time:.2f} seconds")
    print(f"Best RF Parameters: {best_rf_params}")
    print(f"Best RF 5-Fold CV Accuracy: {best_rf_cv_score:.4f} ({best_rf_cv_score*100:.2f}%)")

    # Test Set Evaluation
    t0 = time.perf_counter()
    rf_opt_preds = best_rf_model.predict(X_test_unscaled)
    rf_opt_pred_time = time.perf_counter() - t0

    rf_opt_acc = accuracy_score(y_test, rf_opt_preds)
    rf_opt_prec = precision_score(y_test, rf_opt_preds, average='weighted')
    rf_opt_rec = recall_score(y_test, rf_opt_preds, average='weighted')
    rf_opt_f1 = f1_score(y_test, rf_opt_preds, average='weighted')
    rf_opt_report_dict = classification_report(y_test, rf_opt_preds, target_names=class_names, output_dict=True)
    rf_opt_cm = confusion_matrix(y_test, rf_opt_preds)

    print("\n--- Optimized Random Forest Test Performance ---")
    print(f"Accuracy  : {rf_opt_acc:.4f}")
    print(f"Precision : {rf_opt_prec:.4f}")
    print(f"Recall    : {rf_opt_rec:.4f}")
    print(f"F1-Score  : {rf_opt_f1:.4f}")

    # Save Optimized RF Model
    rf_model_path = os.path.join(output_dir, 'random_forest_optimized.joblib')
    joblib.dump(best_rf_model, rf_model_path)
    print(f"Saved: {rf_model_path}")

    # Optimized RF Feature Importance
    rf_opt_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    rf_fi_csv_path = os.path.join(output_dir, 'optimized_rf_feature_importance.csv')
    rf_opt_importances.to_csv(rf_fi_csv_path, index=False)
    print(f"Saved: {rf_fi_csv_path}")

    # Plot Optimized RF Feature Importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=rf_opt_importances, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('Optimized Random Forest - Feature Importance', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Gini Importance Weight', fontsize=12)
    plt.ylabel('Vibration Feature', fontsize=12)
    for idx, row in rf_opt_importances.iterrows():
        plt.text(row['Importance'] + 0.003, idx, f"{row['Importance']:.4f}", va='center', fontsize=10)
    plt.xlim(0, max(rf_opt_importances['Importance']) * 1.15)
    plt.tight_layout()
    rf_fi_png_path = os.path.join(output_dir, 'optimized_rf_feature_importance.png')
    plt.savefig(rf_fi_png_path)
    plt.close()
    print(f"Saved: {rf_fi_png_path}")


    # ------------------------------------------------------------------------------
    # 3. SVM HYPERPARAMETER TUNING USING PIPELINE (STRICT CV SCALING WITHOUT LEAKAGE)
    # ------------------------------------------------------------------------------
    print("\n=== 3. SVM HYPERPARAMETER TUNING (PIPELINE + GRIDSEARCH CV ON UNSCALED X_TRAIN) ===")

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])

    svm_param_grid = {
        'svm__C': [0.1, 1, 10, 50, 100, 200],
        'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'svm__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }

    print(f"SVM Pipeline Hyperparameter Grid: {svm_param_grid}")

    svm_grid = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=svm_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=1,
        verbose=1
    )

    # Fit strictly on UN-SCALED training data so StandardScaler is fitted inside each CV fold
    t0 = time.perf_counter()
    svm_grid.fit(X_train_unscaled, y_train)
    svm_tuning_time = time.perf_counter() - t0

    best_svm_pipeline = svm_grid.best_estimator_
    # Strip 'svm__' prefix for reporting best hyperparameters
    best_svm_params_raw = svm_grid.best_params_
    best_svm_params = {k.replace('svm__', ''): v for k, v in best_svm_params_raw.items()}
    best_svm_cv_score = float(svm_grid.best_score_)

    print(f"\nSVM Pipeline Tuning Completed in {svm_tuning_time:.2f} seconds")
    print(f"Best SVM Hyperparameters: {best_svm_params}")
    print(f"Best SVM 5-Fold CV Accuracy (Strict Pipeline): {best_svm_cv_score:.4f} ({best_svm_cv_score*100:.2f}%)")

    # Test Set Evaluation (Passing UN-SCALED test data through the pipeline)
    t0 = time.perf_counter()
    svm_opt_preds = best_svm_pipeline.predict(X_test_unscaled)
    svm_opt_pred_time = time.perf_counter() - t0

    svm_opt_acc = accuracy_score(y_test, svm_opt_preds)
    svm_opt_prec = precision_score(y_test, svm_opt_preds, average='weighted')
    svm_opt_rec = recall_score(y_test, svm_opt_preds, average='weighted')
    svm_opt_f1 = f1_score(y_test, svm_opt_preds, average='weighted')
    svm_opt_report_dict = classification_report(y_test, svm_opt_preds, target_names=class_names, output_dict=True)
    svm_opt_cm = confusion_matrix(y_test, svm_opt_preds)

    print("\n--- Optimized SVM (Pipeline) Test Performance ---")
    print(f"Accuracy  : {svm_opt_acc:.4f}")
    print(f"Precision : {svm_opt_prec:.4f}")
    print(f"Recall    : {svm_opt_rec:.4f}")
    print(f"F1-Score  : {svm_opt_f1:.4f}")

    # Save Optimized SVM Pipeline Model
    svm_model_path = os.path.join(output_dir, 'svm_optimized.joblib')
    joblib.dump(best_svm_pipeline, svm_model_path)
    print(f"Saved: {svm_model_path}")


    # ------------------------------------------------------------------------------
    # 4. PLOTS & CONFUSION MATRICES
    # ------------------------------------------------------------------------------
    print("\n=== 4. GENERATING PLOTS & COMPARISONS ===")

    # Individual Confusion Matrices
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(rf_opt_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    ax.set_title('Optimized Random Forest - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Fault Class', fontsize=12)
    ax.set_ylabel('Actual Fault Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    rf_cm_path = os.path.join(output_dir, 'optimized_rf_confusion_matrix.png')
    plt.savefig(rf_cm_path)
    plt.close()
    print(f"Saved: {rf_cm_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(svm_opt_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, cbar=True, ax=ax)
    ax.set_title('Optimized SVM (Pipeline) - Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Fault Class', fontsize=12)
    ax.set_ylabel('Actual Fault Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    svm_cm_path = os.path.join(output_dir, 'optimized_svm_confusion_matrix.png')
    plt.savefig(svm_cm_path)
    plt.close()
    print(f"Saved: {svm_cm_path}")

    # Side-by-Side Confusion Matrix Comparisons
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(rf_baseline_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title(f"Baseline RF (Acc: {rf_baseline_info['metrics']['accuracy']*100:.2f}%)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(rf_opt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
    axes[1].set_title(f"Optimized RF (Acc: {rf_opt_acc*100:.2f}%)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    plt.suptitle('Random Forest: Baseline vs. Optimized Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    rf_comp_cm_path = os.path.join(output_dir, 'rf_baseline_vs_optimized_cm.png')
    plt.savefig(rf_comp_cm_path)
    plt.close()
    print(f"Saved: {rf_comp_cm_path}")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(svm_baseline_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
    axes[0].set_title(f"Baseline SVM (Acc: {svm_baseline_info['metrics']['accuracy']*100:.2f}%)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)

    sns.heatmap(svm_opt_cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
    axes[1].set_title(f"Optimized SVM Pipeline (Acc: {svm_opt_acc*100:.2f}%)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    plt.suptitle('SVM: Baseline vs. Optimized (Strict Pipeline CV) Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    svm_comp_cm_path = os.path.join(output_dir, 'svm_baseline_vs_optimized_cm.png')
    plt.savefig(svm_comp_cm_path)
    plt.close()
    print(f"Saved: {svm_comp_cm_path}")

    # Performance Comparison Table
    comparison_df = pd.DataFrame([
        {
            'Model Stage': 'Baseline Random Forest',
            'Accuracy': rf_baseline_info['metrics']['accuracy'],
            'Precision': rf_baseline_info['metrics']['precision_weighted'],
            'Recall': rf_baseline_info['metrics']['recall_weighted'],
            'F1-score': rf_baseline_info['metrics']['f1_score_weighted'],
            'Best CV Score': 'N/A'
        },
        {
            'Model Stage': 'Optimized Random Forest',
            'Accuracy': rf_opt_acc,
            'Precision': rf_opt_prec,
            'Recall': rf_opt_rec,
            'F1-score': rf_opt_f1,
            'Best CV Score': best_rf_cv_score
        },
        {
            'Model Stage': 'Baseline SVM (RBF)',
            'Accuracy': svm_baseline_info['metrics']['accuracy'],
            'Precision': svm_baseline_info['metrics']['precision_weighted'],
            'Recall': svm_baseline_info['metrics']['recall_weighted'],
            'F1-score': svm_baseline_info['metrics']['f1_score_weighted'],
            'Best CV Score': 'N/A'
        },
        {
            'Model Stage': 'Optimized SVM (Pipeline)',
            'Accuracy': svm_opt_acc,
            'Precision': svm_opt_prec,
            'Recall': svm_opt_rec,
            'F1-score': svm_opt_f1,
            'Best CV Score': best_svm_cv_score
        }
    ])

    summary_csv_path = os.path.join(output_dir, 'tuning_summary.csv')
    comparison_df.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")
    print("\n=== BASELINE VS OPTIMIZED MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))

    # Plot Comparison Bar Chart
    comparison_melted = pd.DataFrame([
        {'Model': 'RF (Baseline)', 'Metric': 'Accuracy', 'Score': rf_baseline_info['metrics']['accuracy']},
        {'Model': 'RF (Baseline)', 'Metric': 'Precision', 'Score': rf_baseline_info['metrics']['precision_weighted']},
        {'Model': 'RF (Baseline)', 'Metric': 'Recall', 'Score': rf_baseline_info['metrics']['recall_weighted']},
        {'Model': 'RF (Baseline)', 'Metric': 'F1-score', 'Score': rf_baseline_info['metrics']['f1_score_weighted']},

        {'Model': 'RF (Optimized)', 'Metric': 'Accuracy', 'Score': rf_opt_acc},
        {'Model': 'RF (Optimized)', 'Metric': 'Precision', 'Score': rf_opt_prec},
        {'Model': 'RF (Optimized)', 'Metric': 'Recall', 'Score': rf_opt_rec},
        {'Model': 'RF (Optimized)', 'Metric': 'F1-score', 'Score': rf_opt_f1},

        {'Model': 'SVM (Baseline)', 'Metric': 'Accuracy', 'Score': svm_baseline_info['metrics']['accuracy']},
        {'Model': 'SVM (Baseline)', 'Metric': 'Precision', 'Score': svm_baseline_info['metrics']['precision_weighted']},
        {'Model': 'SVM (Baseline)', 'Metric': 'Recall', 'Score': svm_baseline_info['metrics']['recall_weighted']},
        {'Model': 'SVM (Baseline)', 'Metric': 'F1-score', 'Score': svm_baseline_info['metrics']['f1_score_weighted']},

        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Accuracy', 'Score': svm_opt_acc},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Precision', 'Score': svm_opt_prec},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'Recall', 'Score': svm_opt_rec},
        {'Model': 'SVM (Optimized Pipeline)', 'Metric': 'F1-score', 'Score': svm_opt_f1},
    ])

    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=comparison_melted, x='Metric', y='Score', hue='Model', palette='tab10')
    plt.title('Phase 5: Baseline vs. Optimized Model Performance Metrics (Strict Pipeline CV)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.ylim(0.85, 1.02)
    for p in chart.patches:
        height = p.get_height()
        if height > 0:
            chart.annotate(f'{height:.4f}',
                           (p.get_x() + p.get_width() / 2., height),
                           ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    chart_png_path = os.path.join(output_dir, 'baseline_vs_optimized_comparison.png')
    plt.savefig(chart_png_path)
    plt.close()
    print(f"Saved: {chart_png_path}")


    # ------------------------------------------------------------------------------
    # 5. SAVE JSON TUNING REPORT
    # ------------------------------------------------------------------------------
    full_report = {
        "project": "Machine Learning-Based Fault Diagnosis of Rotating Machinery",
        "phase": "Phase 5 — Hyperparameter Tuning and Model Optimization",
        "cross_validation": {
            "strategy": "5-Fold Stratified K-Fold",
            "scoring_metric": "accuracy",
            "train_sample_count": len(X_train_unscaled),
            "test_sample_count": len(X_test_unscaled)
        },
        "random_forest": {
            "feature_type": "unscaled",
            "search_grid": rf_param_grid,
            "tuning_time_seconds": rf_tuning_time,
            "best_cv_accuracy": best_rf_cv_score,
            "best_hyperparameters": best_rf_params,
            "baseline_test_metrics": rf_baseline_info['metrics'],
            "optimized_test_metrics": {
                "accuracy": float(rf_opt_acc),
                "precision_weighted": float(rf_opt_prec),
                "recall_weighted": float(rf_opt_rec),
                "f1_score_weighted": float(rf_opt_f1)
            },
            "accuracy_improvement": float(rf_opt_acc - rf_baseline_info['metrics']['accuracy']),
            "f1_improvement": float(rf_opt_f1 - rf_baseline_info['metrics']['f1_score_weighted']),
            "classification_report": rf_opt_report_dict
        },
        "svm": {
            "feature_type": "pipeline_unscaled_input",
            "scaling_inside_cv": "StandardScaler fitted strictly inside each CV fold via Pipeline",
            "search_grid": svm_param_grid,
            "tuning_time_seconds": svm_tuning_time,
            "best_cv_accuracy": best_svm_cv_score,
            "best_hyperparameters": best_svm_params,
            "baseline_test_metrics": svm_baseline_info['metrics'],
            "optimized_test_metrics": {
                "accuracy": float(svm_opt_acc),
                "precision_weighted": float(svm_opt_prec),
                "recall_weighted": float(svm_opt_rec),
                "f1_score_weighted": float(svm_opt_f1)
            },
            "accuracy_improvement": float(svm_opt_acc - svm_baseline_info['metrics']['accuracy']),
            "f1_improvement": float(svm_opt_f1 - svm_baseline_info['metrics']['f1_score_weighted']),
            "classification_report": svm_opt_report_dict
        }
    }

    report_json_path = os.path.join(output_dir, 'tuning_report.json')
    with open(report_json_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f"Saved: {report_json_path}")

    print(f"\nPhase 5 Hyperparameter Tuning & Model Optimization Completed Successfully!")
    print(f"All artifacts saved under: {output_dir}")


if __name__ == '__main__':
    main()
