import os
import json
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

def run_tests():
    print("======================================================================")
    print("RUNNING COMPREHENSIVE TEST SUITE FOR PRESET BEARING SELECTION")
    print("======================================================================")
    
    # 1. Verify Dataset and Assets
    prep_dir = r"d:\FAULT DIAGNOSIS\results\preprocessing"
    model_path = r"d:\FAULT DIAGNOSIS\results\models\tuning\random_forest_optimized.joblib"
    
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    assert os.path.exists(os.path.join(prep_dir, "X_test_unscaled.csv")), "X_test_unscaled.csv not found"
    assert os.path.exists(os.path.join(prep_dir, "y_test.csv")), "y_test.csv not found"
    assert os.path.exists(os.path.join(prep_dir, "label_mapping.json")), "label_mapping.json not found"
    
    X_test = pd.read_csv(os.path.join(prep_dir, "X_test_unscaled.csv"))
    y_test = pd.read_csv(os.path.join(prep_dir, "y_test.csv"))['fault_label'].values
    with open(os.path.join(prep_dir, "label_mapping.json")) as f:
        mapping = json.load(f)
    
    label_map = {int(k): v for k, v in mapping.items()}
    class_names = [label_map[i] for i in range(len(label_map))]
    feature_names = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
    
    print("[PASS] 1. Preprocessing artifacts and model files verified.")
    
    # Compute ground truth first actual sample for each class
    expected_samples = {}
    for c_idx, c_name in label_map.items():
        matches = np.where(y_test == c_idx)[0]
        assert len(matches) > 0, f"No samples found for {c_name}"
        first_row = X_test.iloc[matches[0]]
        expected_samples[c_name] = {f: float(first_row[f]) for f in feature_names}
    
    # 2. Test Streamlit App initialization
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App initialization threw exception: {at.exception}"
    print("[PASS] 2. Streamlit app initialized with zero exceptions.")
    
    # Verify Selectbox Options
    sb = at.selectbox(key="preset_condition_selection")
    assert sb is not None, "Preset selectbox not found by key 'preset_condition_selection'"
    expected_options = [
        "Custom Input",
        "Normal_1",
        "Ball_007_1",
        "Ball_014_1",
        "Ball_021_1",
        "IR_007_1",
        "IR_014_1",
        "IR_021_1",
        "OR_007_6_1",
        "OR_014_6_1",
        "OR_021_6_1"
    ]
    assert sb.options == expected_options, f"Options mismatch! Got: {sb.options}"
    assert sb.value == "Custom Input", f"Default selection should be 'Custom Input', got: {sb.value}"
    print(f"[PASS] 3. Selectbox options verified: {sb.options}")
    
    # Verify initial default values in number inputs
    defaults = {
        'max': 0.2500, 'min': -0.2400, 'mean': 0.0120,
        'sd': 0.0550, 'rms': 0.0560, 'skewness': 0.0200,
        'kurtosis': 3.1000, 'crest': 4.4600, 'form': 4.6600
    }
    for f in feature_names:
        w = at.number_input(key=f"feat_{f}")
        assert w is not None, f"Input widget feat_{f} not found"
        assert abs(w.value - defaults[f]) < 1e-4, f"Initial default mismatch for {f}: expected {defaults[f]}, got {w.value}"
    
    # Verify diagnosis was NOT automatically triggered on initial load
    assert not any("HEALTH STATUS" in str(m.value) or "FAULT DETECTED" in str(m.value) for m in at.markdown), \
        "Diagnosis should not execute on initial load without clicking button"
    print("[PASS] 4. Initial number input defaults verified; diagnosis NOT executed on startup.")
    
    # 3. Test all 10 preset conditions
    test_presets = [
        "Normal_1",
        "Ball_007_1",
        "Ball_014_1",
        "Ball_021_1",
        "IR_007_1",
        "IR_014_1",
        "IR_021_1",
        "OR_007_6_1",
        "OR_014_6_1",
        "OR_021_6_1"
    ]
    
    for preset_name in test_presets:
        # Select preset
        at.selectbox(key="preset_condition_selection").select(preset_name).run()
        assert not at.exception, f"Selecting {preset_name} threw exception: {at.exception}"
        
        # Verify diagnosis did NOT execute upon selecting preset
        assert not any("HEALTH STATUS" in str(m.value) or "FAULT DETECTED" in str(m.value) for m in at.markdown), \
            f"Diagnosis must NOT execute automatically when selecting preset '{preset_name}'!"
        
        # Verify all 9 features are populated with the exact values from test dataset
        expected_sample = expected_samples[preset_name]
        for f in feature_names:
            w_val = at.number_input(key=f"feat_{f}").value
            exp_val = expected_sample[f]
            assert abs(w_val - exp_val) < 1e-4, \
                f"Feature '{f}' mismatch for preset '{preset_name}': expected {exp_val}, got {w_val}"
        
        # Click diagnose button
        btn = at.button[0]
        btn.click().run()
        assert not at.exception, f"Clicking diagnose for {preset_name} threw exception: {at.exception}"
        
        # Verify diagnosis results are displayed
        markdown_texts = " ".join(str(m.value) for m in at.markdown)
        if preset_name == "Normal_1":
            assert "HEALTH STATUS: NORMAL OPERATING CONDITION" in markdown_texts, "Normal_1 should produce NORMAL status banner"
            assert "Normal_1" in markdown_texts, "Normal_1 should appear in prediction text"
        else:
            assert "FAULT DETECTED:" in markdown_texts, f"Fault preset {preset_name} should produce FAULT status banner"
            assert preset_name in markdown_texts, f"{preset_name} should appear in prediction text"
        
        print(f"[PASS] 5. Preset '{preset_name}': correctly populated all 9 features, did not auto-diagnose, and diagnosed correctly on button click.")
    
    # 4. Test Custom Input functionality
    at.selectbox(key="preset_condition_selection").select("Custom Input").run()
    assert not at.exception, "Switching to Custom Input threw exception"
    assert not any("HEALTH STATUS" in str(m.value) or "FAULT DETECTED" in str(m.value) for m in at.markdown), \
        "Diagnosis should not execute upon selecting Custom Input"
    
    # Enter custom values (healthy vibration profile)
    custom_vals = {
        'max': 0.1800, 'min': -0.1900, 'mean': 0.0100,
        'sd': 0.0600, 'rms': 0.0610, 'skewness': -0.1500,
        'kurtosis': 2.9500, 'crest': 2.9500, 'form': 6.1000
    }
    for f, val in custom_vals.items():
        at.number_input(key=f"feat_{f}").set_value(val)
    
    at.run()
    # Click diagnose
    at.button[0].click().run()
    markdown_texts = " ".join(str(m.value) for m in at.markdown)
    assert ("HEALTH STATUS:" in markdown_texts or "FAULT DETECTED:" in markdown_texts), \
        "Custom Input did not trigger diagnosis on button click"
    print("[PASS] 6. Custom Input: manually edited features and diagnosed successfully on button click.")
    
    # 5. Test Physical Validation Rules
    # Violate bound: min >= max
    at.number_input(key="feat_min").set_value(0.5000)
    at.number_input(key="feat_max").set_value(0.2000)
    at.button[0].click().run()
    err_texts = " ".join(str(e.value) for e in at.error)
    assert "Inverted Amplitude Bounds" in err_texts, f"Expected Inverted Amplitude Bounds error, got: {err_texts}"
    print("[PASS] 7. Physical validation: inverted bounds correctly blocked diagnosis.")
    
    print("======================================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! 100% VERIFIED.")
    print("======================================================================")

if __name__ == "__main__":
    run_tests()
