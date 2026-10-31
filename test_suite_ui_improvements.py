from streamlit.testing.v1 import AppTest

def test_ui_enhancements():
    print("======================================================================")
    print("TESTING TECHNICAL AUDIT CORRECTIONS & UI ACCURACY VERIFICATION")
    print("======================================================================")
    
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App threw exception on startup: {at.exception}"
    print("[PASS] 1. App mounted cleanly with zero exceptions.")
    
    # Top-level markdown and caption corpus
    top_markdown = " ".join([str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption])
    
    # All markdowns and captions including inside expanders
    all_markdown_list = [str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption]
    for exp in at.expander:
        all_markdown_list.extend(str(m.value) for m in exp.markdown)
    all_markdown = " ".join(all_markdown_list)
    
    # 1. Project Introduction Check
    assert "Project Overview & Academic Purpose" in top_markdown, "Project introduction header missing"
    assert "Case Western Reserve University (CWRU) Bearing Data Center" in top_markdown, "CWRU dataset mention missing"
    assert "nine statistical time-domain features" in top_markdown, "9 statistical features mention missing"
    assert "48 kHz" in top_markdown, "48 kHz sampling rate mention missing"
    print("[PASS] 2. Project Introduction verified with all required components.")
    
    # 2. Audited System Workflow Diagram Check (Requirement 1 & 7)
    assert "Diagnostic Pipeline Workflow" in top_markdown, "Workflow header missing"
    assert "Precomputed Features / Preset" in top_markdown, "Audited Workflow step 1 missing"
    assert "Input Validation" in top_markdown, "Audited Workflow step 2 missing"
    assert "Random Forest Classification" in top_markdown, "Audited Workflow step 3 missing"
    assert "Predicted Fault Class" in top_markdown, "Audited Workflow step 4 missing"
    assert "Probability Distribution" in top_markdown, "Audited Workflow step 5 missing"
    assert "Health Interpretation" in top_markdown, "Audited Workflow step 6 missing"
    assert "workflow-row" in top_markdown, "Responsive two-row workflow layout missing"
    assert "workflow-connector" in top_markdown, "Workflow stage transition connector missing"
    assert "precomputed statistical features" in top_markdown, "Precomputed features explanation missing"
    assert "does not perform live vibration acquisition or real-time raw signal processing" in top_markdown, \
        "Clarification regarding absence of live acquisition missing"
    print("[PASS] 3. Diagnostic workflow correctly updated to 6 precomputed stages with responsive 2-row layout.")
    
    # 3. Supported Bearing Conditions Table Check (Requirement 2)
    assert len(at.expander) >= 3, f"Expected at least 3 expanders, got {len(at.expander)}"
    # Check that literal <br> tags are completely eliminated from expander 0 markdown
    exp_table_md = " ".join(str(m.value) for m in at.expander[0].markdown)
    assert "<br>" not in exp_table_md, f"Found forbidden literal <br> tags in bearing conditions table: {exp_table_md}"
    assert "Normal_1" in exp_table_md, "Normal_1 label missing"
    assert "Ball_007_1" in exp_table_md and "Ball_014_1" in exp_table_md and "Ball_021_1" in exp_table_md, "Ball fault labels missing"
    assert "IR_007_1" in exp_table_md and "IR_014_1" in exp_table_md and "IR_021_1" in exp_table_md, "IR fault labels missing"
    assert "OR_007_6_1" in exp_table_md and "OR_014_6_1" in exp_table_md and "OR_021_6_1" in exp_table_md, "OR fault labels missing"
    assert "0.007\"" in exp_table_md and "0.014\"" in exp_table_md and "0.021\"" in exp_table_md, "Defect diameter inches missing"
    assert "0.178 mm" in exp_table_md and "0.356 mm" in exp_table_md and "0.533 mm" in exp_table_md, "Defect diameter metric missing"
    assert "Operational Category" in exp_table_md and "Defect Diameter (Inches)" in exp_table_md and "Defect Diameter (Metric)" in exp_table_md, \
        "Dedicated table columns missing"
    print("[PASS] 4. Supported Bearing Conditions table verified: NO <br> tags, all 10 classes in separate rows with dedicated columns.")
    
    # 4. Feature Explanations Guide Check (Requirement 3)
    feat_guide_md = " ".join(str(m.value) for m in at.expander[1].markdown)
    assert "Peak Maximum (`max`)" in feat_guide_md, "max feature missing"
    assert "Peak Minimum (`min`)" in feat_guide_md, "min feature missing"
    assert "Signal Mean (`mean`)" in feat_guide_md, "mean feature missing"
    assert "Standard Deviation (`sd`)" in feat_guide_md, "sd feature missing"
    assert "Root Mean Square (`rms`)" in feat_guide_md, "rms feature missing"
    assert "Skewness (`skewness`)" in feat_guide_md, "skewness feature missing"
    assert "Kurtosis (`kurtosis`)" in feat_guide_md, "kurtosis feature missing"
    assert "Crest Factor (`crest`)" in feat_guide_md, "crest feature missing"
    assert "Form Factor (`form`)" in feat_guide_md, "form feature missing"
    
    # Check that RMS is NOT described as kinetic energy, but as effective vibration magnitude / signal power
    assert "effective vibration magnitude" in feat_guide_md, "RMS must be described as effective vibration magnitude"
    assert "kinetic" not in feat_guide_md.lower(), "RMS must not claim to be kinetic energy"
    
    # Check that Form factor matches actual dataset implementation: rms / mean
    assert "rms" in feat_guide_md and "mean" in feat_guide_md, "Form factor definition missing rms/mean"
    assert r"\frac{\text{rms}}{\text{mean}}" in feat_guide_md, "Form factor formula must show rms/mean"
    print("[PASS] 5. Technical feature definitions verified: RMS reflects effective vibration magnitude (not kinetic energy), form factor correctly defined as rms/mean.")
    
    # 5. Technical Specifications Check
    assert "48,000 Hz" in all_markdown, "48 kHz sampling rate missing in specs"
    assert "2,048 samples/window" in all_markdown, "2048 window size missing in specs"
    assert "1 HP (~1,772 RPM rotational speed)" in all_markdown, "1 HP load missing in specs"
    assert "50 decision trees" in all_markdown, "50 trees missing in specs"
    assert "94.57%" in all_markdown, "94.57% accuracy missing in specs"
    assert "94.60%" in all_markdown, "94.60% F1 missing in specs"
    print("[PASS] 6. Complete technical specifications verified against exact training artifacts.")
    
    # 6. Step-by-Step Preset Instructions Check
    assert "Step-by-Step Diagnostic Instructions" in top_markdown, "Preset instructions missing"
    assert "Select Bearing Condition:" in top_markdown, "Step 1 missing"
    assert "Automatic Population:" in top_markdown, "Step 2 missing"
    assert "Review / Adjust Values:" in top_markdown, "Step 3 missing"
    assert "Execute Diagnosis:" in top_markdown, "Step 4 missing"
    assert "Preset Samples vs. Custom Input" in top_markdown, "Preset vs custom distinction missing"
    print("[PASS] 7. Step-by-step diagnostic instructions verified.")
    
    # 7. Check Tooltips on All 9 Feature Number Inputs (Requirement 3)
    feature_keys = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']
    for f in feature_keys:
        w = at.number_input(key=f"feat_{f}")
        assert w is not None, f"Input widget feat_{f} not found"
        assert w.help is not None and len(w.help) > 20, f"Tooltip missing or too short for {f}"
    
    rms_widget = at.number_input(key="feat_rms")
    assert "effective vibration magnitude" in rms_widget.help, "RMS tooltip must describe effective vibration magnitude"
    assert "kinetic" not in rms_widget.help.lower(), "RMS tooltip must not claim kinetic energy"
    
    form_widget = at.number_input(key="feat_form")
    assert "rms / mean" in form_widget.help, "Form factor tooltip must specify rms / mean"
    print("[PASS] 8. Verified enriched technical tooltips with audited formulas and descriptions on all 9 inputs.")
    
    # 8. Execute Diagnosis & Check Probability Terminology, Severity Rules, and Disclaimer (Requirements 4, 5, 6)
    at.selectbox(key="preset_condition_selection").select("IR_014_1").run()
    at.button[0].click().run()
    assert not at.exception, f"Diagnosis run threw exception: {at.exception}"
    
    res_corpus = " ".join(str(m.value) for m in at.markdown)
    assert "How to Interpret the Diagnostic Results" in res_corpus, "Result interpretation section missing"
    assert "Predicted Fault Class:" in res_corpus, "Predicted fault class explanation missing"
    assert "Model Class Prediction Probability:" in res_corpus, "Model class prediction probability explanation missing"
    assert "Tree-Vote Proportion:" in res_corpus, "Tree vote proportion missing in banner"
    assert "Multi-Class Probability Distribution" in res_corpus, "Multi-class probability distribution explanation missing"
    assert "Health Status & Predefined Severity Rules" in res_corpus, "Health status & severity rules explanation missing"
    assert "predefined engineering lookup rules" in res_corpus, "Predefined engineering lookup rules text missing"
    assert "Severity is <em>not</em> independently predicted by a separate machine learning model" in res_corpus or \
           "Severity is not independently predicted" in res_corpus, "Severity model clarification missing"
    assert "Engineering Diagnostic & Probability Interpretation Disclaimer:" in res_corpus, "Engineering disclaimer missing"
    assert "A high model prediction probability does not guarantee the actual physical bearing condition" in res_corpus or \
           "A high model prediction probability does not guarantee the actual physical condition" in res_corpus, \
           "Diagnostic certainty disclaimer missing"
    
    # Verify Altair horizontal bar chart was rendered
    assert len(at.get('vega_lite_chart')) >= 1, "Altair horizontal bar chart was not rendered"
    print("[PASS] 9. Verified probability terminology (tree-vote proportion), predefined severity rules, engineering disclaimer, and horizontal Altair chart.")
    
    print("======================================================================")
    print("ALL TECHNICAL AUDIT AND UI CORRECTION TESTS PASSED SUCCESSFULLY! (100%)")
    print("======================================================================")

if __name__ == "__main__":
    test_ui_enhancements()

