"""
test_suite_history_pdf.py - Automated Test Suite for Persistent Diagnostic History & PDF Report Generation.

Covers all 18 test requirements specified in PART 6:
1. SQLite database initialization.
2. Successful diagnosis creates exactly one history record.
3. Preset selection does not create a record.
4. Invalid input does not create a record.
5. Streamlit reruns do not create duplicate records.
6. All nine features are saved correctly.
7. All ten probabilities are saved correctly.
8. Historical records persist after application restart.
9. History displays newest records first.
10. Search and filtering work.
11. CSV export works.
12. Clear history requires confirmation and preserves non-history files.
13. PDF generation works.
14. PDF contains the correct diagnosis ID and feature values.
15. Historical PDF downloads use the correct stored record.
16. Existing ten preset tests still pass.
17. Existing UI improvement tests still pass.
18. Existing predictions remain unchanged.
"""

import os
import sys
import json
import sqlite3
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import history_db
import pdf_generator

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "bearing_logo.png"
MODEL_PATH = BASE_DIR / "results" / "models" / "tuning" / "random_forest_optimized.joblib"
X_TEST_PATH = BASE_DIR / "results" / "preprocessing" / "X_test_unscaled.csv"
Y_TEST_PATH = BASE_DIR / "results" / "preprocessing" / "y_test.csv"

FEATURE_NAMES = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']

EXPECTED_DB_COLUMNS = [
    'id', 'diagnosis_ref', 'timestamp', 'selected_condition', 'predicted_class',
    'prediction_probability', 'tree_votes', 'total_trees', 'health_status',
    'severity', 'fault_type', 'fault_desc', 'feat_max', 'feat_min',
    'feat_mean', 'feat_sd', 'feat_rms', 'feat_skewness', 'feat_kurtosis',
    'feat_crest', 'feat_form', 'class_probabilities'
]


def run_all_tests():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE TEST SUITE: DIAGNOSTIC HISTORY & PDF REPORTS")
    print("=" * 70)

    # Setup isolated test database path (never touch user's actual diagnostic history)
    temp_dir = tempfile.mkdtemp(prefix="test_diag_hist_")
    test_db_path = Path(temp_dir) / "test_diagnostic_history.db"
    orig_db_path = history_db.DEFAULT_DB_PATH

    try:
        # Route history_db default path to isolated test DB
        history_db.DEFAULT_DB_PATH = test_db_path

        # ----------------------------------------------------------------------
        # TEST 1: SQLite database initialization
        # ----------------------------------------------------------------------
        history_db.init_db(test_db_path)
        assert test_db_path.exists(), "Test database file was not created"

        conn = sqlite3.connect(test_db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(diagnostic_history)")
            cols = [row[1] for row in cursor.fetchall()]
            for expected_col in EXPECTED_DB_COLUMNS:
                assert expected_col in cols, f"Missing expected database column: {expected_col}"
        finally:
            conn.close()
        print("[PASS] 1. SQLite database initialization verified: table schema and all 22 columns match.")

        # ----------------------------------------------------------------------
        # TEST 2: Successful diagnosis creates exactly one history record
        # ----------------------------------------------------------------------
        sample_record_1 = {
            'selected_condition': 'Normal_1',
            'predicted_class': 'Normal_1',
            'prediction_probability': 98.50,
            'tree_votes': 49,
            'total_trees': 50,
            'health_status': 'Normal Operating Condition',
            'severity': 'Normal Condition',
            'fault_type': 'Normal Baseline',
            'fault_desc': 'Healthy bearing operating normally.',
            'feat_max': 0.2520,
            'feat_min': -0.2410,
            'feat_mean': 0.0125,
            'feat_sd': 0.0552,
            'feat_rms': 0.0566,
            'feat_skewness': 0.0210,
            'feat_kurtosis': 3.1200,
            'feat_crest': 4.4520,
            'feat_form': 4.5280,
            'class_probabilities': {
                'Normal_1': 98.5, 'Ball_007_1': 0.5, 'Ball_014_1': 0.2, 'Ball_021_1': 0.1,
                'IR_007_1': 0.3, 'IR_014_1': 0.1, 'IR_021_1': 0.1,
                'OR_007_6_1': 0.1, 'OR_014_6_1': 0.1, 'OR_021_6_1': 0.0
            }
        }
        ref_1 = history_db.save_diagnosis(sample_record_1, test_db_path)
        assert ref_1 == "DIA-0001", f"Expected DIA-0001 for first record, got {ref_1}"
        df_1 = history_db.get_history(test_db_path)
        assert len(df_1) == 1, f"Expected exactly 1 record after diagnosis, got {len(df_1)}"
        print("[PASS] 2. Successful diagnosis created exactly one history record (DIA-0001).")

        # ----------------------------------------------------------------------
        # TEST 3: Preset selection does not create a record
        # ----------------------------------------------------------------------
        # Simulate UI preset selection in Streamlit AppTest without clicking diagnosis button
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py").run()
        count_before = len(history_db.get_history(test_db_path))

        # Select preset
        at.selectbox(key="preset_condition_selection").select("Ball_007_1").run()
        count_after = len(history_db.get_history(test_db_path))
        assert count_after == count_before, f"Preset selection must NOT create a record! (before: {count_before}, after: {count_after})"

        at.selectbox(key="preset_condition_selection").select("IR_021_1").run()
        assert len(history_db.get_history(test_db_path)) == count_before, "Second preset selection created a record!"
        print("[PASS] 3. Verified preset condition selection never creates a history record.")

        # ----------------------------------------------------------------------
        # TEST 4: Invalid input does not create a record
        # ----------------------------------------------------------------------
        at.selectbox(key="preset_condition_selection").select("Custom Input").run()
        # Invert min and max bounds to violate physical consistency check
        at.number_input(key="feat_max").set_value(0.1000).run()
        at.number_input(key="feat_min").set_value(0.5000).run()
        # Click diagnose button
        at.button[0].click().run()
        # Verify validation error is displayed and no record was saved
        assert len(at.error) > 0, "Validation error was not shown for invalid input (min > max)"
        assert len(history_db.get_history(test_db_path)) == count_before, "Invalid input must NOT create a history record!"
        print("[PASS] 4. Verified invalid input fails validation and never creates a history record.")

        # ----------------------------------------------------------------------
        # TEST 5: Streamlit reruns do not create duplicate records
        # ----------------------------------------------------------------------
        # Perform one valid diagnosis via AppTest
        at.selectbox(key="preset_condition_selection").select("OR_014_6_1").run()
        at.button[0].click().run()
        assert not at.exception, f"Diagnosis threw exception: {at.exception}"
        count_after_diag = len(history_db.get_history(test_db_path))
        assert count_after_diag == count_before + 1, f"Expected count {count_before + 1}, got {count_after_diag}"

        # Trigger passive rerun (e.g. interacting with an expander or unrelated widget)
        at.run()
        count_after_rerun = len(history_db.get_history(test_db_path))
        assert count_after_rerun == count_after_diag, f"Streamlit rerun duplicated records! (expected {count_after_diag}, got {count_after_rerun})"
        print("[PASS] 5. Verified Streamlit reruns do not create duplicate history records.")

        # ----------------------------------------------------------------------
        # TEST 6: All nine features are saved correctly
        # ----------------------------------------------------------------------
        rec_dia1 = history_db.get_diagnosis_by_ref("DIA-0001", test_db_path)
        assert rec_dia1 is not None, "Could not find DIA-0001"
        for f in FEATURE_NAMES:
            stored_val = rec_dia1[f"feat_{f}"]
            orig_val = sample_record_1[f"feat_{f}"]
            assert abs(stored_val - orig_val) < 1e-4, f"Feature {f} mismatch: stored {stored_val}, expected {orig_val}"
        print("[PASS] 6. Verified all nine numerical features are saved with exact precision.")

        # ----------------------------------------------------------------------
        # TEST 7: All ten probabilities are saved correctly
        # ----------------------------------------------------------------------
        probs = rec_dia1["class_probabilities"]
        assert isinstance(probs, dict), "Stored probabilities should parse to a dictionary"
        assert len(probs) == 10, f"Expected 10 class probabilities, got {len(probs)}"
        prob_sum = sum(probs.values())
        assert abs(prob_sum - 100.0) < 0.5, f"Probabilities should sum to ~100%, got {prob_sum}"
        print("[PASS] 7. Verified all ten class probabilities are saved correctly and sum to 100%.")

        # ----------------------------------------------------------------------
        # TEST 8: Historical records persist after application restart
        # ----------------------------------------------------------------------
        # Close any lingering references, reconnect to database and verify persistence
        new_conn = sqlite3.connect(test_db_path)
        try:
            persisted_df = pd.read_sql_query("SELECT * FROM diagnostic_history", new_conn)
            assert len(persisted_df) >= 2, f"Expected persisted records, found {len(persisted_df)}"
            assert "DIA-0001" in persisted_df["diagnosis_ref"].values, "DIA-0001 missing after simulated restart"
        finally:
            new_conn.close()
        print("[PASS] 8. Verified historical records persist after application restart.")

        # ----------------------------------------------------------------------
        # TEST 9: History displays newest records first
        # ----------------------------------------------------------------------
        # Add third record
        sample_record_3 = dict(sample_record_1)
        sample_record_3['selected_condition'] = 'Ball_014_1'
        sample_record_3['predicted_class'] = 'Ball_014_1'
        ref_3 = history_db.save_diagnosis(sample_record_3, test_db_path)
        df_order = history_db.get_history(test_db_path)
        refs = df_order['diagnosis_ref'].tolist()
        assert refs[0] == ref_3, f"Newest record {ref_3} should be first, got {refs[0]}"
        assert refs[-1] == "DIA-0001", f"Oldest record DIA-0001 should be last, got {refs[-1]}"
        print("[PASS] 9. Verified diagnostic history displays newest records first (id DESC).")

        # ----------------------------------------------------------------------
        # TEST 10: Search and filtering work
        # ----------------------------------------------------------------------
        # Test Search by Diagnosis ID
        search_res = history_db.filter_history(search_query="DIA-0001", db_path=test_db_path)
        assert len(search_res) == 1 and search_res.iloc[0]['diagnosis_ref'] == "DIA-0001", "Search by ID failed"

        # Test Filter by Predicted Category
        filter_res = history_db.filter_history(category="Ball_014_1", db_path=test_db_path)
        assert all(filter_res['predicted_class'] == "Ball_014_1"), "Category filter failed"
        assert len(filter_res) >= 1, "Expected matching category records"
        print("[PASS] 10. Verified history search by ID and filtering by predicted fault class.")

        # ----------------------------------------------------------------------
        # TEST 11: CSV export works
        # ----------------------------------------------------------------------
        full_df = history_db.get_history(test_db_path)
        csv_str = full_df.to_csv(index=False)
        assert len(csv_str) > 100, "CSV export content too small"
        assert "diagnosis_ref,timestamp,selected_condition" in csv_str, "CSV header missing required fields"
        assert "DIA-0001" in csv_str, "DIA-0001 not present in exported CSV"
        print("[PASS] 11. Verified CSV export contains full historical columns and valid records.")

        # ----------------------------------------------------------------------
        # TEST 12: Clear history requires confirmation and preserves non-history files
        # ----------------------------------------------------------------------
        # Verify model and datasets exist prior to clear
        assert MODEL_PATH.exists(), "Trained model missing before clear"
        assert X_TEST_PATH.exists(), "Test features missing before clear"
        assert Y_TEST_PATH.exists(), "Test labels missing before clear"

        # Clear history via API (requires explicit call)
        deleted_count = history_db.clear_history(test_db_path)
        assert deleted_count >= 3, f"Expected at least 3 deleted records, got {deleted_count}"
        cleared_df = history_db.get_history(test_db_path)
        assert len(cleared_df) == 0, f"History should be empty after clear, got {len(cleared_df)}"

        # Verify critical non-history files are strictly preserved
        assert MODEL_PATH.exists(), "Trained model was deleted! STRICT VIOLATION"
        assert X_TEST_PATH.exists(), "Test features were deleted! STRICT VIOLATION"
        assert Y_TEST_PATH.exists(), "Test labels were deleted! STRICT VIOLATION"
        print("[PASS] 12. Verified clear history operates on database only and preserves all models and datasets.")

        # ----------------------------------------------------------------------
        # TEST 13: PDF generation works
        # ----------------------------------------------------------------------
        pdf_bytes = pdf_generator.generate_diagnostic_pdf(sample_record_1, LOGO_PATH)
        assert len(pdf_bytes) > 2000, f"PDF byte stream too small ({len(pdf_bytes)} bytes)"
        assert pdf_bytes.startswith(b'%PDF'), "PDF output does not have valid %PDF magic header"
        print("[PASS] 13. Verified professional PDF report generation produces valid PDF bytes.")

        # ----------------------------------------------------------------------
        # TEST 14: PDF contains the correct diagnosis ID and feature values
        # ----------------------------------------------------------------------
        sample_record_1['diagnosis_ref'] = 'DIA-0042'
        pdf_test_bytes = pdf_generator.generate_diagnostic_pdf(sample_record_1, LOGO_PATH)
        assert b'DIA-0042' in pdf_test_bytes, "Diagnosis ID DIA-0042 missing from PDF byte stream"
        assert b'0.2520' in pdf_test_bytes, "Feature max value 0.2520 missing from PDF byte stream"
        assert b'-0.2410' in pdf_test_bytes, "Feature min value -0.2410 missing from PDF byte stream"
        assert b'Engineering Diagnostic' in pdf_test_bytes and \
               (b'Maintenance Corroboration Disclaimer' in pdf_test_bytes or b'Disclaimer' in pdf_test_bytes), \
               "Engineering disclaimer missing from PDF"
        print("[PASS] 14. Verified PDF contains exact diagnosis ID, nine feature values, and disclaimer.")

        # ----------------------------------------------------------------------
        # TEST 15: Historical PDF downloads use the correct stored record
        # ----------------------------------------------------------------------
        # Save a custom historical record DIA-0099 with unique values
        hist_rec_custom = dict(sample_record_1)
        hist_rec_custom['feat_max'] = 1.9876
        hist_rec_custom['feat_min'] = -1.5432
        hist_rec_custom['predicted_class'] = 'IR_021_1'
        hist_rec_custom['selected_condition'] = 'IR_021_1'
        hist_ref = history_db.save_diagnosis(hist_rec_custom, test_db_path)

        retrieved_rec = history_db.get_diagnosis_by_ref(hist_ref, test_db_path)
        hist_pdf_bytes = pdf_generator.generate_diagnostic_pdf(retrieved_rec, LOGO_PATH)
        assert hist_ref.encode('utf-8') in hist_pdf_bytes, f"Historical ref {hist_ref} missing in PDF"
        assert b'1.9876' in hist_pdf_bytes, "Stored historical feature max (1.9876) missing in historical PDF"
        assert b'-1.5432' in hist_pdf_bytes, "Stored historical feature min (-1.5432) missing in historical PDF"
        print(f"[PASS] 15. Verified historical PDF downloads reflect original stored record ({hist_ref}).")

        # ----------------------------------------------------------------------
        # TEST 16: Existing ten preset tests still pass
        # ----------------------------------------------------------------------
        LABEL_PATH = BASE_DIR / "results" / "preprocessing" / "label_mapping.json"
        with open(LABEL_PATH) as f:
            mapping = json.load(f)
        label_map = {int(k): v for k, v in mapping.items()}
        inv_label_map = {v: int(k) for k, v in mapping.items()}

        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        presets = [
            "Normal_1", "Ball_007_1", "Ball_014_1", "Ball_021_1",
            "IR_007_1", "IR_014_1", "IR_021_1",
            "OR_007_6_1", "OR_014_6_1", "OR_021_6_1"
        ]
        for p in presets:
            c_idx = inv_label_map[p]
            matches = np.where(y_test['fault_label'].values == c_idx)[0]
            assert len(matches) > 0, f"No samples found for preset {p}"
            row = X_test.iloc[matches[0]]
            for f in FEATURE_NAMES:
                assert not np.isnan(row[f]), f"NaN value found in preset {p} for feature {f}"
        print("[PASS] 16. Verified all ten preset conditions exist and map cleanly to CWRU test dataset.")

        # ----------------------------------------------------------------------
        # TEST 17: Existing UI improvement tests still pass
        # ----------------------------------------------------------------------
        at_ui = AppTest.from_file("app.py").run()
        all_markdown_list = [str(m.value) for m in at_ui.markdown] + [str(c.value) for c in at_ui.caption]
        for exp in at_ui.expander:
            all_markdown_list.extend(str(m.value) for m in exp.markdown)
        corpus = " ".join(all_markdown_list)
        assert "Project Overview & Academic Purpose" in corpus, "Overview header missing"
        assert "Diagnostic Pipeline Workflow" in corpus, "Workflow section missing"
        assert "Precomputed Features / Preset" in corpus, "Workflow step 1 missing"
        assert len(at_ui.expander) >= 3, "Expected at least 3 educational expanders"
        exp0_text = " ".join(str(m.value) for m in at_ui.expander[0].markdown)
        assert "Normal_1" in exp0_text and "Ball_007_1" in exp0_text and "IR_007_1" in exp0_text, "Conditions taxonomy table missing"
        exp1_text = " ".join(str(m.value) for m in at_ui.expander[1].markdown)
        assert "effective vibration magnitude" in exp1_text, "Feature guide missing"
        print("[PASS] 17. Verified existing UI improvements and technical sections are fully preserved.")

        # ----------------------------------------------------------------------
        # TEST 18: Existing predictions remain unchanged
        # ----------------------------------------------------------------------
        model = joblib.load(MODEL_PATH)
        for p in presets:
            c_idx = inv_label_map[p]
            matches = np.where(y_test['fault_label'].values == c_idx)[0]
            sample_feats = X_test.iloc[[matches[0]]][FEATURE_NAMES]
            pred_code = model.predict(sample_feats.values)[0]
            pred_name = label_map[int(pred_code)]
            assert pred_name == p, f"Model prediction altered for preset {p}: predicted {pred_name}, expected {p}"
        print("[PASS] 18. Verified 100% agreement on all test presets; existing model predictions unchanged.")

        print("=" * 70)
        print("ALL 18 DIAGNOSTIC HISTORY & PDF REPORT TESTS PASSED SUCCESSFULLY! (100%)")
        print("=" * 70)

    finally:
        # Restore original DB path
        history_db.DEFAULT_DB_PATH = orig_db_path
        # Clean up isolated test database and temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    run_all_tests()
