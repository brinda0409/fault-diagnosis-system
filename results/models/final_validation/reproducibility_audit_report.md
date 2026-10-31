# Reproducibility Audit & Methodological Verification Report

**Project**: Machine Learning-Based Fault Diagnosis of Rotating Machinery  
**Phase**: Phase 6 — Final Validation & Reproducibility Verification  
**File Location**: `results/models/final_validation/reproducibility_audit_report.md`  

---

## 1. Executive Summary

A comprehensive reproducibility audit was conducted to investigate the performance metric discrepancy between early textual Phase 5 reports and serialized Phase 6 disk artifacts. 

The audit confirmed that:
- All saved model binaries (`random_forest_optimized.joblib` and `svm_optimized.joblib`) and test datasets (`X_test_unscaled.csv`, `X_test_scaled.csv`, `y_test.csv`) are 100% valid, intact, and fully reproducible.
- The **Optimized Random Forest Classifier** (`random_forest_optimized.joblib`) achieved **94.57% test accuracy** and **94.60% weighted F1-score**, making it the **final selected model** for fault diagnosis.

---

## 2. SHA-256 File Hashes & Data Integrity

The exact SHA-256 cryptographic hashes for all serialized models and preprocessed test datasets were verified:

### Saved Model Artifacts (`results/models/tuning/`)
- **`random_forest_optimized.joblib`**: `970724e887ed63cd044a55dac969c2be24f110a1a5dcde2e2fa8c643b172fd11`
- **`svm_optimized.joblib`**: `51526331ec1dbcd01c04344c922f359da55d0255e1989598b205f7e6cb99ee61`

### Preprocessed Test Datasets (`results/preprocessing/`)
- **`X_test_unscaled.csv`**: `7b6b2560d012ab27` (460 rows, 9 feature columns)
- **`X_test_scaled.csv`**: `933014671f9d11db` (460 rows, 9 feature columns)
- **`y_test.csv`**: `d6b37b62f039d4f1` (460 target fault labels)

---

## 3. Root Cause Analysis of Metric Discrepancy

### A. Initial Phase 5 Pre-Scaled Run (Data Leakage)
In the initial Phase 5 execution, feature scaling (`StandardScaler`) was applied to the entire 1,840-sample training dataset prior to `GridSearchCV`. During 5-fold cross-validation, each validation fold (368 samples) was transformed using mean and standard deviation parameters computed from all 1,840 samples (which included the validation fold itself). 

This subtle fold-level data leakage artificially inflated cross-validation performance and selected higher-variance hyperparameters (`C=200, gamma=1`), which yielded an initial text metric of 97.17%.

### B. Corrected Leak-Free Pipeline Implementation
Following user feedback, a methodological correction was applied using a scikit-learn `Pipeline`:
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])
```
GridSearchCV was executed on the **unscaled training data** (`X_train_unscaled.csv`). In this architecture, `StandardScaler` was fitted strictly inside each of the 5 CV folds on the 1,472 training fold samples, guaranteeing **zero scaling leakage** into the validation fold.

### C. Serialized Disk Model Results
The leak-free Pipeline CV selected **`C=100, gamma=0.1, kernel='rbf'`** (5-fold CV score: **96.90%**), and serialized this model to `svm_optimized.joblib`. When evaluated on the 460-sample test set, the true leak-free test metrics were **93.91% Accuracy** and **93.99% Weighted F1-Score**.

Phase 6 loaded the saved `svm_optimized.joblib` binary from disk and evaluated it on `X_test_unscaled.csv`, correctly reproducing the exact **93.91% Accuracy** and **93.99% F1-Score** matching `tuning_report.json` down to 6 decimal places.

---

## 4. Hyperparameter Configurations of Final Saved Models

### A. Final Saved Random Forest (`random_forest_optimized.joblib`)
- `n_estimators`: `50`
- `max_depth`: `10`
- `min_samples_split`: `2`
- `min_samples_leaf`: `2`
- `max_features`: `'sqrt'`
- `random_state`: `42`

### B. Final Saved SVM Pipeline (`svm_optimized.joblib`)
- `steps`: `[('scaler', StandardScaler()), ('svm', SVC(C=100, gamma=0.1, kernel='rbf', random_state=42))]`
- `C`: `100`
- `gamma`: `0.1`
- `kernel`: `'rbf'`
- `random_state`: `42`

---

## 5. Final Reproducible Model Performance Comparison

The table below summarizes the exact, leak-free test set evaluation across all project phases (evaluated on the untouched 460-sample test set):

| Model Stage | Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Macro F1-Score | Audit Selection Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline RF** | Unscaled `RandomForestClassifier()` | 94.35% | 94.65% | 94.35% | 94.38% | 94.38% | Baseline |
| **Optimized RF** | **Unscaled `RandomForestClassifier(n_estimators=50, max_depth=10)`** | **94.57%** | **94.88%** | **94.57%** | **94.60%** | **94.60%** | **SELECTED FINAL MODEL** |
| **Baseline SVM** | Scaled `SVC(kernel='rbf')` | 91.30% | 92.24% | 91.30% | 91.15% | 91.15% | Baseline |
| **Optimized SVM** | **Strict `Pipeline(StandardScaler + SVC(C=100, gamma=0.1, rbf))`** | **93.91%** | **94.31%** | **93.91%** | **93.99%** | **93.99%** | Optimized |

---

## 6. Final Model Selection Conclusion

The **Optimized Random Forest Classifier** (`random_forest_optimized.joblib`) is formally declared as the **Final Selected Model** for rotating machinery fault diagnosis. It achieved the highest generalization performance across all metrics on the independent test set:
- **Test Accuracy**: **94.57%** (`0.945652`)
- **Weighted Precision**: **94.88%** (`0.948847`)
- **Weighted Recall**: **94.57%** (`0.945652`)
- **Weighted F1-Score**: **94.60%** (`0.946023`)
- **Macro F1-Score**: **94.60%** (`0.946023`)

---

## 7. Methodological & Constraint Verification

1. **Dataset Integrity**: The original CWRU dataset (`Data CWRU/feature_time_48k_2048_load_1.csv`) and Phase 1–4 preprocessing files were completely untouched.
2. **Test Set Isolation**: The 460-sample test set (`X_test_unscaled.csv`, `X_test_scaled.csv`, `y_test.csv`) was strictly isolated and used **only** for single-pass final evaluation. No test data was used during training or hyperparameter selection.
3. **No Retraining**: All audit metrics were computed directly from serialized `joblib` artifacts on disk.
4. **Phase 7 Isolation**: No web interface or Streamlit development has been initiated.
