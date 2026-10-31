import os
import json
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

import history_db
import pdf_generator

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Rotating Machinery Fault Diagnosis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern predictive-maintenance aesthetic
st.markdown("""
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Banner */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 25px;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .main-title {
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Result Diagnostic Banners */
    .status-normal {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        color: #ecfdf5;
        margin-bottom: 20px;
    }
    .status-fault {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        color: #fef2f2;
        margin-bottom: 20px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 15px;
        margin-bottom: 15px;
        border-bottom: 2px solid #334155;
        padding-bottom: 8px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
        transform: translateY(-1px);
    }

    /* Project Introduction Box */
    .intro-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 15px;
        font-size: 0.96rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Workflow Pipeline */
    .workflow-container {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    .workflow-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }
    .workflow-connector {
        text-align: center;
        padding: 6px 0;
        color: #38bdf8;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .wf-step {
        flex: 1 1 0px;
        background: #0f172a;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
        min-width: 0;
    }
    .wf-num {
        display: inline-block;
        width: 22px;
        height: 22px;
        line-height: 22px;
        background: #2563eb;
        color: white;
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .wf-title {
        color: #f1f5f9;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 3px;
        line-height: 1.25;
    }
    .wf-desc {
        color: #94a3b8;
        font-size: 0.73rem;
        line-height: 1.3;
    }
    .wf-arrow {
        color: #38bdf8;
        font-size: 1.2rem;
        font-weight: bold;
        flex: 0 0 auto;
        padding: 0 4px;
    }

    /* Instruction Card */
    .instruction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 15px;
    }

    /* Explanation Card */
    .explain-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
    .explain-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .explain-body {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.45;
    }

    /* Diagnostic Disclaimer Card */
    .disclaimer-card {
        background-color: #1e293b;
        border: 1px solid #475569;
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 15px;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. PATHS & MODEL LOADING (CACHED FOR PERFORMANCE)
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "bearing_logo.png"

MODEL_PATH = str(BASE_DIR / "results" / "models" / "tuning" / "random_forest_optimized.joblib")
LABEL_PATH = str(BASE_DIR / "results" / "preprocessing" / "label_mapping.json")
PREP_DIR = str(BASE_DIR / "results" / "preprocessing")

FEATURE_NAMES = ['max', 'min', 'mean', 'sd', 'rms', 'skewness', 'kurtosis', 'crest', 'form']

HUMAN_INTERPRETATION = {
    'Normal_1': {
        'type': 'Normal Condition',
        'desc': 'Healthy bearing operating in normal condition (1 HP motor load).',
        'severity': 'NORMAL',
        'color': '#10b981'
    },
    'Ball_007_1': {
        'type': 'Rolling Element (Ball) Fault',
        'desc': 'Minor rolling element flaw (0.007" defect diameter under 1 HP load).',
        'severity': 'FAULT — MODERATE',
        'color': '#f59e0b'
    },
    'Ball_014_1': {
        'type': 'Rolling Element (Ball) Fault',
        'desc': 'Medium rolling element flaw (0.014" defect diameter under 1 HP load).',
        'severity': 'FAULT — HIGH',
        'color': '#ef4444'
    },
    'Ball_021_1': {
        'type': 'Rolling Element (Ball) Fault',
        'desc': 'Severe rolling element flaw (0.021" defect diameter under 1 HP load).',
        'severity': 'FAULT — CRITICAL',
        'color': '#dc2626'
    },
    'IR_007_1': {
        'type': 'Inner Race Fault',
        'desc': 'Minor inner raceway flaw (0.007" defect diameter under 1 HP load).',
        'severity': 'FAULT — MODERATE',
        'color': '#f59e0b'
    },
    'IR_014_1': {
        'type': 'Inner Race Fault',
        'desc': 'Medium inner raceway flaw (0.014" defect diameter under 1 HP load).',
        'severity': 'FAULT — HIGH',
        'color': '#ef4444'
    },
    'IR_021_1': {
        'type': 'Inner Race Fault',
        'desc': 'Severe inner raceway flaw (0.021" defect diameter under 1 HP load).',
        'severity': 'FAULT — CRITICAL',
        'color': '#dc2626'
    },
    'OR_007_6_1': {
        'type': 'Outer Race Fault',
        'desc': 'Minor outer raceway flaw (0.007" defect diameter, 6:00 load position, 1 HP load).',
        'severity': 'FAULT — MODERATE',
        'color': '#f59e0b'
    },
    'OR_014_6_1': {
        'type': 'Outer Race Fault',
        'desc': 'Medium outer raceway flaw (0.014" defect diameter, 6:00 load position, 1 HP load).',
        'severity': 'FAULT — HIGH',
        'color': '#ef4444'
    },
    'OR_021_6_1': {
        'type': 'Outer Race Fault',
        'desc': 'Severe outer raceway flaw (0.021" defect diameter, 6:00 load position, 1 HP load).',
        'severity': 'FAULT — CRITICAL',
        'color': '#dc2626'
    }
}


@st.cache_resource
def load_assets():
    """Load optimized model and label mapping with error handling."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    if not os.path.exists(LABEL_PATH):
        st.error(f"Label mapping file not found at: {LABEL_PATH}")
        st.stop()

    model = joblib.load(MODEL_PATH)
    
    with open(LABEL_PATH, 'r') as f:
        raw_mapping = json.load(f)
    
    # Map int index -> class name
    label_map = {int(k): v for k, v in raw_mapping.items()}
    class_names = [label_map[i] for i in range(len(label_map))]
    
    return model, label_map, class_names


@st.cache_data
def load_preset_samples():
    """Load real representative sample rows from test dataset for quick testing."""
    unscaled_path = os.path.join(PREP_DIR, 'X_test_unscaled.csv')
    y_test_path = os.path.join(PREP_DIR, 'y_test.csv')
    
    if os.path.exists(unscaled_path) and os.path.exists(y_test_path):
        try:
            X_test = pd.read_csv(unscaled_path)
            y_test = pd.read_csv(y_test_path)['fault_label'].values
            return X_test, y_test
        except Exception:
            return None, None
    return None, None


model, label_mapping, class_names = load_assets()


@st.cache_data
def get_preset_samples_map():
    """Retrieve actual representative sample row from CWRU test dataset for each of the 10 classes."""
    X_test, y_test = load_preset_samples()
    if X_test is None or y_test is None:
        return {}
    
    presets = {}
    for class_idx, class_name in label_mapping.items():
        matches = np.where(y_test == class_idx)[0]
        if len(matches) > 0:
            first_idx = matches[0]
            row = X_test.iloc[first_idx]
            presets[class_name] = {col: float(row[col]) for col in FEATURE_NAMES}
    return presets


# Baseline default feature values
DEFAULT_VALS = {
    'max': 0.2500,
    'min': -0.2400,
    'mean': 0.0120,
    'sd': 0.0550,
    'rms': 0.0560,
    'skewness': 0.0200,
    'kurtosis': 3.1000,
    'crest': 4.4600,
    'form': 4.6600
}

# 10 predefined bearing condition options from CWRU dataset + Custom Input
PRESET_OPTIONS = [
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

# Initialize session state for feature inputs if not already set
for f_name in FEATURE_NAMES:
    if f"feat_{f_name}" not in st.session_state:
        st.session_state[f"feat_{f_name}"] = float(DEFAULT_VALS[f_name])


def on_preset_change():
    """Callback when preset condition dropdown is changed."""
    st.session_state.pop("active_diagnosis_record", None)
    selected = st.session_state.get('preset_condition_selection', 'Custom Input')
    if selected == "Custom Input":
        for f in FEATURE_NAMES:
            st.session_state[f"feat_{f}"] = float(DEFAULT_VALS[f])
    else:
        preset_map = get_preset_samples_map()
        if selected in preset_map:
            sample = preset_map[selected]
            for f in FEATURE_NAMES:
                st.session_state[f"feat_{f}"] = float(sample[f])


# ------------------------------------------------------------------------------
# 3. SIDEBAR: SYSTEM INFORMATION & DIAGNOSTIC SPECS
# ------------------------------------------------------------------------------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=75)
    st.title("System Specs")
    
    st.markdown("""
    **Project**: Rotating Machinery Fault Diagnosis  
    **Dataset**: Case Western Reserve Univ (CWRU)  
    **Sampling Rate**: 48 kHz | 2048 pts/window  
    **Motor Load**: 1 HP  
    """)
    
    st.markdown("---")
    st.subheader("🤖 Active ML Model")
    st.success("Optimized Random Forest Classifier")
    st.markdown("""
    - **Test Accuracy**: **94.57%**
    - **Weighted F1**: **94.60%**
    - **Trees (`n_estimators`)**: 50
    - **Max Depth**: 10
    """)
    
    st.markdown("---")
    st.subheader("⚡ Preset Conditions")
    st.markdown("""
    Use the **Select Bearing Condition (Preset Samples)** dropdown on the main dashboard to load real vibration profiles for any of the 10 CWRU bearing conditions.
    """)


# ------------------------------------------------------------------------------
# 4. MAIN PAGE CONTENT
# ------------------------------------------------------------------------------
# Header Banner & Project Introduction
st.markdown("""
<div class="main-header">
    <div class="main-title">⚙️ Predictive Maintenance Fault Diagnosis App</div>
    <div class="main-subtitle">Automated Machine Health Monitoring & Vibration Time-Domain Feature Analysis</div>
    <div class="intro-box">
        <strong>Project Overview & Academic Purpose:</strong> This intelligent predictive maintenance platform demonstrates automated health monitoring and fault diagnosis for rolling element bearings in industrial rotating machinery. Utilizing supervised machine learning, the system classifies bearing operational health from <strong>nine statistical time-domain features</strong> extracted from vibration acceleration signals sampled at 48 kHz. All training data and performance benchmarks originate from the standardized <strong>Case Western Reserve University (CWRU) Bearing Data Center</strong> experimental records operating under a 1 HP motor load.
    </div>
</div>
""", unsafe_allow_html=True)

# Overview Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown('<div class="metric-card"><div class="metric-val">10</div><div class="metric-label">Fault Classes</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-card"><div class="metric-val">9</div><div class="metric-label">Vibration Features</div></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-card"><div class="metric-val">94.57%</div><div class="metric-label">Diagnostic Accuracy</div></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown('<div class="metric-card"><div class="metric-val">1 HP</div><div class="metric-label">Operating Load</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 5. DIAGNOSTIC PIPELINE WORKFLOW & SYSTEM ARCHITECTURE
# ------------------------------------------------------------------------------
st.markdown('<div class="section-header">🔄 Diagnostic Pipeline Workflow</div>', unsafe_allow_html=True)
st.caption("Operational classification and health interpretation workflow implemented in this application:")

st.markdown("""
<div class="workflow-container">
    <div class="workflow-row">
        <div class="wf-step">
            <div class="wf-num">1</div>
            <div class="wf-title">Precomputed Features / Preset</div>
            <div class="wf-desc">9 Statistical Features<br>CWRU Test Preset or Custom</div>
        </div>
        <div class="wf-arrow">➔</div>
        <div class="wf-step">
            <div class="wf-num">2</div>
            <div class="wf-title">Input Validation</div>
            <div class="wf-desc">Type & Range Checks<br>Physical Consistency Rules</div>
        </div>
        <div class="wf-arrow">➔</div>
        <div class="wf-step">
            <div class="wf-num">3</div>
            <div class="wf-title">Random Forest Classification</div>
            <div class="wf-desc">50 Decision Trees<br>Ensemble Feature Splitting</div>
        </div>
    </div>
    <div class="workflow-connector">
        ⬇ Pipeline Execution Flow
    </div>
    <div class="workflow-row">
        <div class="wf-step">
            <div class="wf-num">4</div>
            <div class="wf-title">Predicted Fault Class</div>
            <div class="wf-desc">10-Class Classification<br>Ensemble Consensus Class</div>
        </div>
        <div class="wf-arrow">➔</div>
        <div class="wf-step">
            <div class="wf-num">5</div>
            <div class="wf-title">Probability Distribution</div>
            <div class="wf-desc">Tree-Vote Proportions<br>Multi-Class Likelihood Spread</div>
        </div>
        <div class="wf-arrow">➔</div>
        <div class="wf-step">
            <div class="wf-num">6</div>
            <div class="wf-title">Health Interpretation</div>
            <div class="wf-desc">Predefined Severity Rules<br>Maintenance Guidance</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("ℹ️ **Data Processing Architecture Note:** This deployed application accepts precomputed statistical features or CWRU benchmark preset samples. It does not perform live vibration acquisition or real-time raw signal processing. Time-domain feature extraction (segmenting 48 kHz acceleration signals into 2048-point windows and extracting nine statistical metrics) was performed during offline dataset preparation.")

# Compact Educational Expanders
with st.expander("📖 Supported Bearing Conditions & Defect Taxonomy (10 Classes)", expanded=False):
    st.markdown("""
    The diagnostic engine distinguishes between **ten distinct operational conditions** across four primary bearing health categories from the standardized Case Western Reserve University (CWRU) 1 HP dataset. Flaws include electro-discharge machining (EDM) localized defects with diameters of **0.007"** (0.178 mm), **0.014"** (0.356 mm), and **0.021"** (0.533 mm):
    
    | Operational Category | Fault Mechanism & Location | Defect Diameter (Inches) | Defect Diameter (Metric) | Trained Class Label | Predefined Severity |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Normal Baseline** | Healthy bearing operating under standard nominal conditions | None (0.000") | None (0.000 mm) | `Normal_1` | Normal Condition |
    | **Rolling Element (Ball)** | Localized EDM spall on spherical rolling element | 0.007" (7 mils) | 0.178 mm | `Ball_007_1` | Moderate Fault |
    | **Rolling Element (Ball)** | Localized EDM spall on spherical rolling element | 0.014" (14 mils) | 0.356 mm | `Ball_014_1` | High Fault |
    | **Rolling Element (Ball)** | Localized EDM spall on spherical rolling element | 0.021" (21 mils) | 0.533 mm | `Ball_021_1` | Critical Fault |
    | **Inner Raceway (IR)** | Localized EDM spall on rotating inner ring | 0.007" (7 mils) | 0.178 mm | `IR_007_1` | Moderate Fault |
    | **Inner Raceway (IR)** | Localized EDM spall on rotating inner ring | 0.014" (14 mils) | 0.356 mm | `IR_014_1` | High Fault |
    | **Inner Raceway (IR)** | Localized EDM spall on rotating inner ring | 0.021" (21 mils) | 0.533 mm | `IR_021_1` | Critical Fault |
    | **Outer Raceway (OR)** | Stationary outer ring flaw at 6:00 (centered in load zone) | 0.007" (7 mils) | 0.178 mm | `OR_007_6_1` | Moderate Fault |
    | **Outer Raceway (OR)** | Stationary outer ring flaw at 6:00 (centered in load zone) | 0.014" (14 mils) | 0.356 mm | `OR_014_6_1` | High Fault |
    | **Outer Raceway (OR)** | Stationary outer ring flaw at 6:00 (centered in load zone) | 0.021" (21 mils) | 0.533 mm | `OR_021_6_1` | Critical Fault |
    """)

with st.expander("📐 Technical Feature Guide: Nine Time-Domain Features Explained", expanded=False):
    st.markdown(r"""
    Time-domain statistical features capture instantaneous shocks, dynamic vibration energy, and waveform shape alterations caused by bearing flaws passing over raceways. Below are the verified mathematical formulations extracted over 2048-point windows:
    
    1. **Peak Maximum (`max`)**: Highest positive instantaneous acceleration amplitude within the window ($\max x_i$, in **g**). Indicates peak outward impact shock magnitude.
    2. **Peak Minimum (`min`)**: Largest negative instantaneous acceleration amplitude within the window ($\min x_i$, in **g**). Quantifies maximum inward rebound acceleration.
    3. **Signal Mean (`mean`)**: Arithmetic average of acceleration ($\mu = \frac{1}{N}\sum_{i=1}^N x_i$, in **g**). Captures unidirectional DC offset or transducer bias (nominally near zero for dynamic vibration).
    4. **Standard Deviation (`sd`)**: Statistical dispersion about the mean ($\sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \mu)^2}$, in **g**). Quantifies dynamic alternating vibration power around the mean.
    5. **Root Mean Square (`rms`)**: Quadratic mean of signal ($x_{\text{rms}} = \sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$, in **g RMS**). Represents **effective vibration magnitude** and steady-state signal power; widely used in vibration standards (e.g., ISO 10816/20816) as the primary indicator of overall mechanical vibration severity.
    6. **Skewness (`skewness`)**: Third standardized central moment ($\frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^3}{\sigma^3}$, Dimensionless). Quantifies signal distribution asymmetry around the mean; non-zero values reflect directional impact bias.
    7. **Kurtosis (`kurtosis`)**: Fourth standardized central moment ($\frac{\frac{1}{N}\sum_{i=1}^N (x_i - \mu)^4}{\sigma^4}$, Dimensionless). Measures impulsiveness and peakedness of the signal. Healthy signals exhibit Gaussian kurtosis ($\approx 3.0$), while recurring impact spalls produce elevated peakedness ($> 3.0$).
    8. **Crest Factor (`crest`)**: Peak-to-RMS ratio ($\text{Crest} = \frac{\max(x_i)}{x_{\text{rms}}} = \frac{\text{max}}{\text{rms}}$, Dimensionless). Highly sensitive to early-stage localized spalling where sharp impulsive shock peaks appear before overall RMS magnitude increases.
    9. **Form Factor (`form`)**: Ratio of RMS vibration magnitude to arithmetic mean ($\text{Form} = \frac{x_{\text{rms}}}{\mu} = \frac{\text{rms}}{\text{mean}}$, Dimensionless). Characterizes waveform profile alterations and energy concentration relative to mean acceleration. *(Note: In the CWRU dataset feature extraction, this is mathematically computed as $\text{rms} / \text{mean}$).*
    """)

with st.expander("🛠️ Complete Technical Specifications & Model Architecture", expanded=False):
    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.markdown("""
        ##### 📊 Dataset & Sensor Setup
        - **Source**: Case Western Reserve University (CWRU) Bearing Data Center
        - **Motor Load**: 1 HP (~1,772 RPM rotational speed)
        - **Sampling Rate**: 48,000 Hz (48 kHz drive-end accelerometer)
        - **Sampling Window**: 2,048 samples/window (~42.67 ms per instance)
        - **Dataset Size**: 2,300 total windowed instances (10 classes × 230 samples)
        - **Data Split**: Stratified 80% Train (1,840 samples) / 20% Test (460 samples)
        - **Input Dimensions**: 9 statistical features (`max`, `min`, `mean`, `sd`, `rms`, `skewness`, `kurtosis`, `crest`, `form`)
        """)
    with tech_col2:
        st.markdown("""
        ##### 🤖 Classifier & Validation Metrics
        - **Algorithm**: Optimized Random Forest Classifier (`random_forest_optimized.joblib`)
        - **Number of Estimators**: 50 decision trees
        - **Maximum Tree Depth**: 10
        - **Min Samples Split / Leaf**: 2 / 2
        - **Max Features**: `'sqrt'` (3 features per split)
        - **Untouched Test Accuracy**: **94.57%** (435/460 correct predictions)
        - **Weighted Precision**: **94.88%**
        - **Weighted Recall**: **94.57%**
        - **Weighted F1-Score**: **94.60%**
        """)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 6. PRESET BEARING CONDITION SELECTION
# ------------------------------------------------------------------------------
st.markdown('<div class="section-header">⚙️ Select Bearing Condition (Preset Samples)</div>', unsafe_allow_html=True)
st.caption("Select a predefined bearing condition profile to automatically load actual test sample measurements, or choose 'Custom Input' for manual specification.")

# Step-by-Step Instructions
st.markdown("""
<div class="instruction-card">
    <div style="font-weight: 600; color: #38bdf8; margin-bottom: 6px;">📋 Step-by-Step Diagnostic Instructions:</div>
    <ol style="margin: 0; padding-left: 20px; line-height: 1.6; color: #e2e8f0; font-size: 0.92rem;">
        <li><strong>Select Bearing Condition:</strong> Choose a preset condition profile from the dropdown below (or choose <em>Custom Input</em>).</li>
        <li><strong>Automatic Population:</strong> The nine numerical feature fields populate automatically with real test measurements from the CWRU dataset.</li>
        <li><strong>Review / Adjust Values:</strong> Inspect the amplitude bounds, dynamic energy, and waveform shape parameters (or edit values for exploratory testing).</li>
        <li><strong>Execute Diagnosis:</strong> Click the <strong>Diagnose Bearing Fault Condition</strong> button to trigger Random Forest model inference and view diagnostic results.</li>
    </ol>
    <div style="margin-top: 8px; font-size: 0.85rem; color: #94a3b8;">
        💡 <strong>Preset Samples vs. Custom Input:</strong> Presets load verified ground-truth test data rows. <em>Custom Input</em> mode allows manual entry of all nine vibration features for scenario simulations and exploratory sensitivity tests.
    </div>
</div>
""", unsafe_allow_html=True)

selected_preset = st.selectbox(
    "Select Bearing Condition (Preset Samples):",
    options=PRESET_OPTIONS,
    key="preset_condition_selection",
    on_change=on_preset_change,
    help="Select a CWRU test dataset sample profile to auto-populate all 9 vibration features, or choose 'Custom Input' to enter features manually."
)

preset_map = get_preset_samples_map()
if selected_preset != "Custom Input":
    if selected_preset in preset_map:
        interp = HUMAN_INTERPRETATION.get(selected_preset, {})
        desc = interp.get('desc', 'CWRU test sample')
        st.info(f"📥 **Preset Loaded:** `{selected_preset}` — {desc} | All 9 features populated from CWRU test data. Click **Diagnose Bearing Fault Condition** below to evaluate.")
    else:
        st.error(f"⚠️ Test sample for '{selected_preset}' could not be loaded: test dataset files are missing from `{PREP_DIR}`.")
else:
    st.info("✏️ **Custom Input Mode Active:** Enter time-domain vibration features manually in the fields below, then click **Diagnose Bearing Fault Condition**.")

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 7. FEATURE INPUT LAYOUT
# ------------------------------------------------------------------------------
st.markdown('<div class="section-header">📊 Enter Vibration Signal Features</div>', unsafe_allow_html=True)
st.caption("Provide time-domain statistical features extracted from 12k/48k acceleration sampling windows (2048 pts).")

input_container = st.container()

with input_container:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Amplitude Bounds & Baseline")
        max_val = st.number_input("Peak Maximum (`max`)", key="feat_max", format="%.4f", help="Highest positive instantaneous acceleration amplitude in the 2048-point window (Unit: g). Indicates maximum outward shock magnitude.")
        min_val = st.number_input("Peak Minimum (`min`)", key="feat_min", format="%.4f", help="Largest negative instantaneous acceleration amplitude in the 2048-point window (Unit: g). Indicates maximum reverse impact excursion.")
        mean_val = st.number_input("Signal Mean (`mean`)", key="feat_mean", format="%.4f", help="Arithmetic mean of signal acceleration (Unit: g): (1/N) * sum(x_i). Detects sensor DC offset or unidirectional baseline shifts (nominally near zero).")
        
    with col2:
        st.markdown("##### Dynamic Signal Energy")
        sd_val = st.number_input("Standard Deviation (`sd`)", key="feat_sd", format="%.4f", help="Standard deviation of vibration (Unit: g): sqrt((1/(N-1)) * sum((x_i - mean)^2)). Quantifies dynamic alternating vibration power around the mean.")
        rms_val = st.number_input("Root Mean Square (`rms`)", key="feat_rms", format="%.4f", help="Root Mean Square acceleration (Unit: g RMS): sqrt((1/N) * sum(x_i^2)). Measures effective vibration magnitude and signal power; the primary baseline metric for overall mechanical vibration severity.")
        skew_val = st.number_input("Skewness (`skewness`)", key="feat_skewness", format="%.4f", help="Third standardized statistical moment (Dimensionless): (1/N) * sum((x_i - mean)^3) / sd^3. Measures waveform asymmetry; non-zero indicates directional impact asymmetry.")
        
    with col3:
        st.markdown("##### Waveform Shape Parameters")
        kurt_val = st.number_input("Kurtosis (`kurtosis`)", key="feat_kurtosis", format="%.4f", help="Fourth standardized statistical moment (Dimensionless): (1/N) * sum((x_i - mean)^4) / sd^4. Quantifies signal impulsiveness/peakedness (Normal Gaussian ~ 3.0; faulty bearings exhibit sharp impact spikes > 3.0).")
        crest_val = st.number_input("Crest Factor (`crest`)", key="feat_crest", format="%.4f", help="Ratio of peak acceleration to RMS magnitude (Dimensionless): max / rms. Sensitive to early-stage impact spalls and impulsive shock peaks before overall RMS rises.")
        form_val = st.number_input("Form Factor (`form`)", key="feat_form", format="%.4f", help="Ratio of RMS vibration magnitude to arithmetic mean (Dimensionless): rms / mean. Characterizes waveform shape profile and relative vibration energy distribution.")

st.markdown("<br>", unsafe_allow_html=True)

# Diagnose Button & Prediction Trigger
diagnose_clicked = st.button("🔍 Diagnose Bearing Fault Condition", width="stretch")

if diagnose_clicked:
    # --------------------------------------------------------------------------
    # 5. INPUT VALIDATION & PREDICTION
    # --------------------------------------------------------------------------
    input_dict = {
        'max': max_val,
        'min': min_val,
        'mean': mean_val,
        'sd': sd_val,
        'rms': rms_val,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'crest': crest_val,
        'form': form_val
    }
    
    # 1. Structural and Numerical Validation
    missing_keys = [k for k in FEATURE_NAMES if k not in input_dict]
    invalid_type_fields = []
    non_finite_fields = []

    for k in FEATURE_NAMES:
        if k in input_dict:
            v = input_dict[k]
            # In Python, bool is a subclass of int (isinstance(True, int) is True),
            # so explicitly reject booleans as inappropriate for continuous vibration features.
            if isinstance(v, bool) or not isinstance(v, (int, float, np.number)):
                invalid_type_fields.append(k)
            elif v is None or not np.isfinite(v):
                non_finite_fields.append(k)

    # 2. Physical Domain Consistency Validation (enforced only when numeric & finite)
    physical_errors = []
    if not missing_keys and not invalid_type_fields and not non_finite_fields:
        min_val_f = float(input_dict['min'])
        max_val_f = float(input_dict['max'])
        sd_val_f = float(input_dict['sd'])
        rms_val_f = float(input_dict['rms'])

        if min_val_f >= max_val_f:
            physical_errors.append(
                f"Inverted Amplitude Bounds: Peak minimum ({min_val_f:.4f}g) cannot be greater than or equal to peak maximum ({max_val_f:.4f}g)."
            )
        if sd_val_f < 0.0:
            physical_errors.append(
                f"Negative Dispersion: Standard deviation ({sd_val_f:.4f}g) cannot be negative."
            )
        if rms_val_f < 0.0:
            physical_errors.append(
                f"Negative Signal Energy: Root Mean Square energy ({rms_val_f:.4f}g) cannot be negative."
            )

    # 3. Decision Gate: Block inference if any validation or physical error occurred
    if missing_keys:
        st.session_state.pop("active_diagnosis_record", None)
        st.error(f"Missing Feature Error: The following required features are missing: {', '.join(missing_keys)}.")
    elif invalid_type_fields:
        st.session_state.pop("active_diagnosis_record", None)
        st.error(f"Data Type Error: Non-numeric values (or booleans) detected in: {', '.join(invalid_type_fields)}. All vibration features must be real numbers.")
    elif non_finite_fields:
        st.session_state.pop("active_diagnosis_record", None)
        st.error(f"Numerical Error: Non-finite values (None, NaN, Inf, -Inf) detected in: {', '.join(non_finite_fields)}. Please enter finite real numbers.")
    elif physical_errors:
        st.session_state.pop("active_diagnosis_record", None)
        for err in physical_errors:
            st.error(f"Physical Consistency Error: {err}")
        st.warning("Diagnosis Blocked: Inference was not executed because the signal features violate physical reality.")
    else:
        # Construct DataFrame preserving exact trained column order
        input_df = pd.DataFrame([input_dict])[FEATURE_NAMES]
        
        # Predict class and probabilities
        pred_idx = int(model.predict(input_df)[0])
        pred_probas = model.predict_proba(input_df)[0]
        
        pred_class = label_mapping[pred_idx]
        confidence = float(pred_probas[pred_idx] * 100)
        
        # Calculate actual tree votes independently from individual estimators
        tree_predictions = [int(tree.predict(input_df.values)[0]) for tree in model.estimators_]
        actual_tree_votes = int(tree_predictions.count(pred_idx))
        total_tree_count = int(len(model.estimators_))
        
        interp = HUMAN_INTERPRETATION.get(pred_class, {
            'type': 'Unknown Class',
            'desc': 'Fault class detected.',
            'severity': 'UNKNOWN',
            'color': '#64748b'
        })
        
        # Prepare complete diagnosis record
        rec_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'selected_condition': selected_preset,
            'predicted_class': pred_class,
            'prediction_probability': confidence,
            'tree_votes': actual_tree_votes,
            'total_trees': total_tree_count,
            'health_status': "Normal Operating Condition" if pred_class == "Normal_1" else f"Fault Condition ({interp['type']})",
            'severity': interp['severity'],
            'fault_type': interp['type'],
            'fault_desc': interp['desc'],
            'feat_max': float(max_val),
            'feat_min': float(min_val),
            'feat_mean': float(mean_val),
            'feat_sd': float(sd_val),
            'feat_rms': float(rms_val),
            'feat_skewness': float(skew_val),
            'feat_kurtosis': float(kurt_val),
            'feat_crest': float(crest_val),
            'feat_form': float(form_val),
            'class_probabilities': {c: float(p * 100) for c, p in zip(class_names, pred_probas)}
        }
        
        # Atomic persistent storage to local SQLite database
        diagnosis_ref = history_db.save_diagnosis(rec_data)
        rec_data['diagnosis_ref'] = diagnosis_ref
        
        # Cache active diagnosis record in session state for persistence across rerun downloads
        st.session_state["active_diagnosis_record"] = rec_data

# ------------------------------------------------------------------------------
# 8. DIAGNOSTIC RESULTS DISPLAY & DOWNLOADABLE PDF REPORT
# ------------------------------------------------------------------------------
active_res = st.session_state.get("active_diagnosis_record")

if active_res:
    pred_class = active_res["predicted_class"]
    confidence = active_res["prediction_probability"]
    tree_votes = active_res.get("tree_votes", 0)
    total_trees = active_res.get("total_trees", 50)
    diag_ref = active_res.get("diagnosis_ref", "DIA-0001")
    
    interp = HUMAN_INTERPRETATION.get(pred_class, {
        'type': 'Unknown Class',
        'desc': 'Fault class detected.',
        'severity': 'UNKNOWN',
        'color': '#64748b'
    })
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📋 Diagnostic Results & Health Status</div>', unsafe_allow_html=True)
    
    # Diagnostic Result Banner
    if pred_class == "Normal_1":
        st.markdown(f"""
        <div class="status-normal">
            <h2 style="margin:0; color:#10b981; font-weight:700;">✅ HEALTH STATUS: NORMAL OPERATING CONDITION</h2>
            <p style="font-size:1.1rem; margin-top:8px; margin-bottom:5px;"><strong>Diagnosis ID:</strong> <code>{diag_ref}</code> | <strong>Predicted Class:</strong> <code>{pred_class}</code> | <strong>Class Prediction Probability:</strong> {confidence:.2f}% (Tree-Vote Proportion: {tree_votes}/{total_trees})</p>
            <p style="margin:0; color:#a7f3d0;">{interp['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-fault">
            <h2 style="margin:0; color:#f87171; font-weight:700;">⚠️ FAULT DETECTED: {interp['type'].upper()}</h2>
            <p style="font-size:1.1rem; margin-top:8px; margin-bottom:5px;"><strong>Diagnosis ID:</strong> <code>{diag_ref}</code> | <strong>Predicted Class:</strong> <code>{pred_class}</code> | <strong>Assigned Severity:</strong> <span style="color:#fca5a5; font-weight:700;">{interp['severity']}</span> | <strong>Class Prediction Probability:</strong> {confidence:.2f}% (Tree-Vote Proportion: {tree_votes}/{total_trees})</p>
            <p style="margin:0; color:#fecaca;">{interp['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Probability Distribution & Feature Impact
    res_col1, res_col2 = st.columns([1.3, 1])
    
    with res_col1:
        st.markdown("##### 📊 Multi-Class Prediction Probability Distribution")
        
        # Prepare dataframe with all 10 classes, sorted descending by probability
        probs_dict = active_res["class_probabilities"]
        proba_df = pd.DataFrame({
            'Fault Class': list(probs_dict.keys()),
            'Probability (%)': list(probs_dict.values()),
            'Is Predicted': [c == pred_class for c in probs_dict.keys()]
        }).sort_values(by='Probability (%)', ascending=False)
        
        highlight_color = '#10b981' if pred_class == 'Normal_1' else '#38bdf8'
        
        # Horizontal bar chart displaying all 10 classes without label clipping
        bars = alt.Chart(proba_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            y=alt.Y(
                'Fault Class:N',
                sort=None,
                title=None,
                axis=alt.Axis(labelColor='#e2e8f0', labelFontSize=12, labelLimit=250)
            ),
            x=alt.X(
                'Probability (%):Q',
                title='Model Class Prediction Probability (%)',
                axis=alt.Axis(labelColor='#94a3b8', titleColor='#94a3b8', gridColor='#334155'),
                scale=alt.Scale(domain=[0, 105])
            ),
            color=alt.condition(
                alt.datum['Is Predicted'],
                alt.value(highlight_color),
                alt.value('#334155')
            ),
            tooltip=[
                alt.Tooltip('Fault Class:N', title='Class'),
                alt.Tooltip('Probability (%):Q', title='Probability (%)', format='.2f')
            ]
        )
        
        text = bars.mark_text(
            align='left',
            baseline='middle',
            dx=6,
            color='#f8fafc',
            fontSize=11,
            fontWeight='bold'
        ).encode(
            text=alt.Text('Probability (%):Q', format='.1f')
        )
        
        chart = (bars + text).properties(height=360)
        st.altair_chart(chart, width="stretch")
        st.caption("Horizontal bar chart displaying all 10 fault classes sorted by descending probability. The predicted class is highlighted.")
        
    with res_col2:
        st.markdown("##### 💡 Input Feature Summary")
        input_summary_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Input Value': [f"{float(active_res[f'feat_{f}']):.4f}" for f in FEATURE_NAMES]
        })
        st.dataframe(input_summary_df, width="stretch", hide_index=True, height=360)

    # --------------------------------------------------------------------------
    # DOWNLOAD DIAGNOSTIC REPORT (PDF) BUTTON
    # --------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    pdf_report_bytes = pdf_generator.generate_diagnostic_pdf(active_res, LOGO_PATH)
    pdf_filename = f"Bearing_Diagnostic_{diag_ref}.pdf"
    
    st.download_button(
        label=f"📥 Download Diagnostic Report (PDF) — {diag_ref}",
        data=pdf_report_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        width="stretch",
        key="btn_download_current_pdf_report"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # DIAGNOSTIC RESULT EXPLANATION & ENGINEERING INTERPRETATION
    # --------------------------------------------------------------------------
    st.markdown("##### 🔍 How to Interpret the Diagnostic Results")
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown(f"""
        <div class="explain-card">
            <div class="explain-title">🎯 Predicted Fault Class: <code>{pred_class}</code></div>
            <div class="explain-body">
                The Random Forest classifier maps the nine statistical features to the winning candidate bearing condition. In this evaluation, the signal profile matches <strong>{interp['type']}</strong> ({interp['desc']}).
            </div>
        </div>
        <div class="explain-card">
            <div class="explain-title">📈 Model Class Prediction Probability: {confidence:.2f}%</div>
            <div class="explain-body">
                Calculated via <code>predict_proba()</code> as the model's posterior class prediction probability. Independently verified across individual decision tree estimators: exactly <strong>{tree_votes} of {total_trees} trees</strong> voted directly for <code>{pred_class}</code>. This reflects algorithmic feature partition consensus, but is <em>not</em> calibrated physical fault certainty. A high prediction probability does not guarantee the actual physical bearing condition.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with exp_col2:
        st.markdown(f"""
        <div class="explain-card">
            <div class="explain-title">📊 Multi-Class Probability Distribution</div>
            <div class="explain-body">
                The horizontal bar chart displays the relative probability score across all ten classes sorted by descending likelihood, highlighting the winning class. A single dominant bar indicates clear ensemble agreement, whereas distributed probabilities indicate boundary proximity between adjacent fault sizes.
            </div>
        </div>
        <div class="explain-card">
            <div class="explain-title">🛡️ Health Status & Predefined Severity Rules</div>
            <div class="explain-body">
                The machine learning model predicts the fault class. Assigned severity (<span style="color:{interp['color']}; font-weight:700;">{interp['severity']}</span>) is determined using <strong>predefined engineering lookup rules</strong> based on the predicted defect size (0.007" = Moderate, 0.014" = High, 0.021" = Critical; Baseline = Normal). Severity is <em>not</em> independently predicted by a separate machine learning model.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Engineering Diagnostic Disclaimer
    st.markdown("""
    <div class="disclaimer-card">
        <strong>⚠️ Engineering Diagnostic & Probability Interpretation Disclaimer:</strong> The reported model prediction probability reflects the proportion of decision trees voting for a class (via <code>predict_proba()</code>) based on historical CWRU benchmark test data. It represents algorithmic class assignment probability, not calibrated physical fault certainty. A high model prediction probability does not guarantee the actual physical condition of the bearing. In industrial predictive maintenance, automated predictions should always be corroborated with vibration spectral analysis (FFT and envelope demodulation), acoustic emission, thermography, and lubrication debris analysis before scheduling intrusive maintenance or replacement. Severity interpretation is mapped using predefined engineering rules corresponding to physical defect dimensions.
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 9. DIAGNOSTIC HISTORY & PREVIOUS PREDICTIONS
# ------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📜 Diagnostic History & Previous Predictions</div>', unsafe_allow_html=True)
st.caption("Locally stored historical diagnoses in SQLite. Search, filter, inspect details, export CSV, or download previous PDF reports:")

# Fetch complete history to check presence of records
all_history_df = history_db.get_history()

if all_history_df.empty:
    st.info("ℹ️ No diagnostic history records found yet. Execute a bearing diagnosis above to automatically record and view previous evaluations.")
else:
    # Filter and Search Controls
    hist_search_col, hist_filter_col = st.columns([1.2, 1.2])
    
    with hist_search_col:
        search_query = st.text_input(
            "🔍 Search by Diagnosis ID",
            placeholder="e.g. DIA-0001",
            key="hist_search_input",
            help="Enter all or part of a diagnosis reference code (e.g. DIA-0001) to search records."
        )
        
    with hist_filter_col:
        category_options = ["All Classes"] + list(HUMAN_INTERPRETATION.keys())
        selected_filter_cat = st.selectbox(
            "🎯 Filter by Predicted Fault Category",
            options=category_options,
            key="hist_category_filter",
            help="Filter diagnostic history table by the predicted bearing fault class."
        )
        
    # Query filtered records
    filtered_history = history_db.filter_history(
        category=selected_filter_cat,
        search_query=search_query
    )
    
    if filtered_history.empty:
        st.warning("⚠️ No diagnostic history records match your search or filter criteria.")
    else:
        # Display clean table with required columns:
        # Diagnosis ID, Date and time, Selected condition, Predicted fault, Prediction probability, Health status
        display_df = filtered_history[[
            'diagnosis_ref', 'timestamp', 'selected_condition',
            'predicted_class', 'prediction_probability', 'health_status'
        ]].copy()
        
        display_df.rename(columns={
            'diagnosis_ref': 'Diagnosis ID',
            'timestamp': 'Date and Time',
            'selected_condition': 'Selected Condition',
            'predicted_class': 'Predicted Fault',
            'prediction_probability': 'Prediction Probability (%)',
            'health_status': 'Health Status'
        }, inplace=True)
        display_df['Prediction Probability (%)'] = display_df['Prediction Probability (%)'].map(lambda p: f"{float(p):.2f}%")
        
        st.dataframe(display_df, width="stretch", hide_index=True)
    
    # Download Complete History as CSV
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = all_history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Complete History (CSV)",
        data=csv_bytes,
        file_name=f"Bearing_Diagnostic_History_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        width="stretch",
        key="btn_download_history_csv"
    )
    
    # Expander: View Details of Selected Diagnosis & Download Historical PDF Report
    with st.expander("🔍 Inspect Historical Record Details & Download Historical PDF", expanded=False):
        ref_options = filtered_history['diagnosis_ref'].tolist() if not filtered_history.empty else all_history_df['diagnosis_ref'].tolist()
        selected_hist_ref = st.selectbox(
            "Select Diagnosis Record to Inspect:",
            options=ref_options,
            key="selected_hist_ref_inspect"
        )
        if selected_hist_ref:
            hist_record = history_db.get_diagnosis_by_ref(selected_hist_ref)
            if hist_record:
                det_c1, det_c2 = st.columns(2)
                with det_c1:
                    clean_status = str(hist_record['health_status']).replace("FAULT DETECTED:", "Fault Condition:").replace("NORMAL OPERATING CONDITION", "Normal Condition")
                    st.markdown(f"""
                    **Diagnosis Reference:** `{hist_record['diagnosis_ref']}`  
                    **Date & Time:** {hist_record['timestamp']}  
                    **Selected Profile:** {hist_record['selected_condition']}  
                    **Predicted Class:** `{hist_record['predicted_class']}`  
                    **Model Prediction Probability:** {hist_record['prediction_probability']:.2f}%  
                    **Tree Votes:** {hist_record['tree_votes']} / {hist_record['total_trees']} trees  
                    **Severity:** `{hist_record['severity']}`  
                    **Health Status:** {clean_status}  
                    """)
                with det_c2:
                    # Download Historical PDF using original stored values
                    hist_pdf = pdf_generator.generate_diagnostic_pdf(hist_record, LOGO_PATH)
                    st.download_button(
                        label=f"📥 Download Historical Report (PDF) — {selected_hist_ref}",
                        data=hist_pdf,
                        file_name=f"Bearing_Diagnostic_{selected_hist_ref}.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key=f"dl_hist_pdf_{selected_hist_ref}"
                    )
                    st.caption("ℹ️ This PDF report is generated strictly from the stored historical database record, preserving original values.")
                
                # Show stored feature values for this historical record
                st.markdown("###### Stored Feature Values for this Historical Record")
                hist_feats = {
                    'Feature': FEATURE_NAMES,
                    'Stored Value': [f"{float(hist_record.get(f'feat_{f}', 0.0)):.4f}" for f in FEATURE_NAMES]
                }
                st.dataframe(pd.DataFrame(hist_feats), width="stretch", hide_index=True)

    # Expander: Clear History with Confirmation
    with st.expander("🗑️ Clear Diagnostic History", expanded=False):
        st.warning("⚠️ **Warning:** Clearing history permanently deletes all historical diagnostic records from the local SQLite database. This action cannot be undone.")
        confirm_clear = st.checkbox(
            "I confirm that I want to permanently delete all stored diagnostic history records.",
            key="confirm_clear_history_checkbox"
        )
        if st.button("Permanently Clear History", disabled=not confirm_clear, key="btn_confirm_clear_history"):
            deleted_count = history_db.clear_history()
            st.session_state.pop("active_diagnosis_record", None)
            st.success(f"Successfully cleared {deleted_count} diagnostic records from database.")
            st.rerun()

st.markdown("---")
st.caption("Machine Learning-Based Fault Diagnosis of Rotating Machinery | Undergraduate Project | Case Western Reserve University (CWRU) Bearing Dataset")
