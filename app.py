# ================================================================
# NeuroScan AI — Alzheimer's Prediction App
# Features: Single Predict · Batch Upload · Dashboard · History · Download
# Run with: streamlit run app.py
# ================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
import warnings
from datetime import datetime, date
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Alzheimer's Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp { background: #F8F9FC; }
header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E8EBF0;
    box-shadow: 2px 0 12px rgba(0,0,0,0.04);
}
section[data-testid="stSidebar"] * { color: #374151 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8EBF0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label {
    color: #6B7280 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #111827 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #FFFFFF !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 8px !important;
    color: #111827 !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid #D1D5DB !important;
    border-radius: 8px !important;
    color: #111827 !important;
}

/* ── Slider ── */
.stSlider > div > div > div {
    background: #E5E7EB !important;
}
.stSlider > div > div > div > div {
    background: #6366F1 !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #FFFFFF !important;
    color: #059669 !important;
    border: 1.5px solid #059669 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #F0FDF4 !important;
    border-color: #047857 !important;
    color: #047857 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 2px solid #E8EBF0;
    gap: 0;
    padding: 0 4px;
    border-radius: 12px 12px 0 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    border-radius: 0 !important;
    padding: 14px 22px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #6366F1 !important;
    background: #F5F3FF !important;
}
.stTabs [aria-selected="true"] {
    color: #6366F1 !important;
    border-bottom: 3px solid #6366F1 !important;
    font-weight: 600 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #FFFFFF;
    border: 1px solid #E8EBF0;
    border-top: none;
    border-radius: 0 0 12px 12px;
    padding: 24px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #E8EBF0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #F9FAFB !important;
    border: 1px solid #E8EBF0 !important;
    border-radius: 8px !important;
    color: #374151 !important;
    font-weight: 500 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #FAFAFA !important;
    border: 2px dashed #D1D5DB !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
}

/* ── Divider ── */
hr { border-color: #E8EBF0 !important; border-width: 1px !important; }

/* ── Text ── */
h1, h2, h3 { color: #111827 !important; }
p { color: #374151 !important; }
label { color: #374151 !important; }

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 10px !important;
    border: none !important;
}
[data-testid="stInfo"] {
    background: #EEF2FF !important;
    color: #4338CA !important;
    border-left: 4px solid #6366F1 !important;
}
[data-testid="stSuccess"] {
    background: #F0FDF4 !important;
    color: #166534 !important;
    border-left: 4px solid #22C55E !important;
}
[data-testid="stError"] {
    background: #FFF1F2 !important;
    color: #9F1239 !important;
    border-left: 4px solid #F43F5E !important;
}

/* ── Custom components ── */
.result-card {
    border-radius: 16px;
    padding: 28px 32px;
    margin: 12px 0;
    text-align: center;
}
.card-ad {
    background: linear-gradient(135deg, #FFF1F2, #FEE2E2);
    border: 1.5px solid #FCA5A5;
}
.card-healthy {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 1.5px solid #86EFAC;
}
.info-box {
    background: #FFFFFF;
    border: 1px solid #E8EBF0;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.powerbi-box {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border: 1.5px solid #FCD34D;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
}
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E8EBF0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin: 16px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ════════════════════════════════════════════════════════
HISTORY_FILE = "prediction_history.xlsx"
FEATURES     = ['ADL', 'FunctionalAssessment', 'MMSE',
                'MemoryComplaints', 'BehavioralProblems']
HEALTHY_RANGES = {
    'MMSE'                : (24, 30),
    'FunctionalAssessment': (7,  10),
    'ADL'                 : (7,  10),
    'MemoryComplaints'    : (0,   0),
    'BehavioralProblems'  : (0,   0),
}
MAX_VALS = {'ADL': 10, 'FunctionalAssessment': 10,
            'MMSE': 30, 'MemoryComplaints': 1,
            'BehavioralProblems': 1}


def risk_label(prob):
    if prob >= 0.70: return "High Risk",   "#f43f5e"
    if prob >= 0.40: return "Medium Risk", "#f59e0b"
    return "Low Risk", "#10b981"


def load_history():
    if os.path.exists(HISTORY_FILE):
        return pd.read_excel(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "timestamp", "patient_id", "adl", "functional_assessment",
        "mmse", "memory_complaints", "behavioral_problems",
        "probability", "risk_level", "prediction"
    ])


def append_history(row: dict):
    df  = load_history()
    df  = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_excel(HISTORY_FILE, index=False)


def to_excel_bytes(df: pd.DataFrame, sheet="Results") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=False, sheet_name=sheet)
        ws = w.sheets[sheet]
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 40)
    return buf.getvalue()


def dark_fig(w=6, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F9FAFB')
    return fig, ax


def style_ax(ax):
    ax.tick_params(colors='#6B7280', labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, axis='y', alpha=0.5, color='#E5E7EB', lw=0.8)


# ════════════════════════════════════════════════════════
# LOAD MODEL
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load('alzheimers_disease_model.pkl')


pkg       = load_model()
model     = pkg['model']
scaler    = pkg['scaler']
features  = pkg['features']
threshold = pkg['threshold']
perf      = pkg['performance']
feat_imp  = pkg['feature_importance']


def predict_single(vals: dict) -> tuple[float, int]:
    vec    = np.array([[vals[f] for f in features]])
    scaled = scaler.transform(vec)
    prob   = model.predict_proba(scaled)[0, 1]
    return float(prob), int(prob >= threshold)


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        'adl'                  : 'ADL',
        'functionalassessment' : 'FunctionalAssessment',
        'mmse'                 : 'MMSE',
        'memorycomplaints'     : 'MemoryComplaints',
        'behavioralproblems'   : 'BehavioralProblems',
    }
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_')
                  for c in df.columns]
    df.rename(columns=col_map, inplace=True)

    X      = df[features].values
    scaled = scaler.transform(X)
    probs  = model.predict_proba(scaled)[:, 1]

    df['probability'] = probs.round(4)
    df['risk_level']  = pd.cut(
        probs,
        bins=[-0.001, 0.40, 0.70, 1.001],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    df['prediction'] = pd.Series(
        (probs >= threshold).astype(int)
    ).map({0: "No Alzheimer's", 1: "Alzheimer's Detected"}).values
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


# ════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:24px 0 16px;'>
        <div style='width:56px;height:56px;background:linear-gradient(135deg,#6366F1,#8B5CF6);
                    border-radius:16px;display:inline-flex;align-items:center;
                    justify-content:center;font-size:28px;margin-bottom:12px;
                    box-shadow:0 4px 14px rgba(99,102,241,0.35);'>🧠</div>
        <h2 style='font-family:Playfair Display,serif; color:#111827;
                   font-size:20px; margin:0 0 4px; font-weight:600;'>
            NeuroScan AI
        </h2>
        <p style='color:#9CA3AF; font-size:11px; margin:0;
                  letter-spacing:0.08em; text-transform:uppercase;
                  font-weight:600;'>
            Alzheimer's Detection
        </p>
    </div>
    <hr style='border-color:#E8EBF0; margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:0.08em;color:#6B7280;margin:0 0 10px;'>
        Model Performance
    </p>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{perf['test_acc']*100:.1f}%")
    c1.metric("F1 Score", f"{perf['f1']*100:.1f}%")
    c2.metric("Recall",   f"{perf['recall']*100:.1f}%")
    c2.metric("AUC",      f"{perf['auc']:.3f}")

    gap_c  = "#059669" if abs(perf['gap']) < 0.05 else "#D97706"
    gap_bg = "#F0FDF4" if abs(perf['gap']) < 0.05 else "#FFFBEB"
    gap_br = "#86EFAC" if abs(perf['gap']) < 0.05 else "#FCD34D"
    st.markdown(f"""
    <div style='background:{gap_bg}; border:1px solid {gap_br};
                border-radius:10px; padding:12px 14px; margin:12px 0;'>
        <p style='margin:0; color:#6B7280; font-size:11px; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.06em;'>
            Train-Test Gap
        </p>
        <p style='margin:4px 0 2px; font-size:20px; font-weight:700;
                  color:{gap_c};'>{perf['gap']:+.4f}</p>
        <p style='margin:0; color:{gap_c}; font-size:12px; font-weight:500;'>
            {'✅ No overfitting' if abs(perf['gap'])<0.05
             else '⚠️ Slight overfit'}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E8EBF0;'>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:0.08em;color:#6B7280;margin:0 0 10px;'>
        Model Info
    </p>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:13px; line-height:2.2;'>
        <span style='color:#9CA3AF;'>Algorithm</span><br>
        <span style='color:#111827;font-weight:600;'>
            {pkg.get('model_name','SVM')}</span><br>
        <span style='color:#9CA3AF;'>Features</span><br>
        <span style='color:#111827;font-weight:600;'>
            {len(features)} cognitive markers</span><br>
        <span style='color:#9CA3AF;'>Balancing</span><br>
        <span style='color:#111827;font-weight:600;'>SMOTE</span><br>
        <span style='color:#9CA3AF;'>Threshold</span><br>
        <span style='color:#111827;font-weight:600;'>{threshold}</span><br>
        <span style='color:#9CA3AF;'>Version</span><br>
        <span style='color:#111827;font-weight:600;'>
            {pkg.get('version','v1.0')}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#E8EBF0;'>", unsafe_allow_html=True)

    hist = load_history()
    if not hist.empty:
        st.markdown("""
        <p style='font-size:11px;font-weight:600;text-transform:uppercase;
                  letter-spacing:0.08em;color:#6B7280;margin:0 0 10px;'>
            History Stats
        </p>""", unsafe_allow_html=True)
        n_pos = len(hist[hist.prediction.str.contains("Alzheimer", na=False)]) \
                if 'prediction' in hist.columns else 0
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;'>
            <div style='background:#F5F3FF;border:1px solid #DDD6FE;
                        border-radius:8px;padding:10px;text-align:center;'>
                <div style='font-size:18px;font-weight:700;color:#6366F1;'>
                    {len(hist)}</div>
                <div style='font-size:10px;color:#7C3AED;font-weight:600;'>
                    Total</div>
            </div>
            <div style='background:#FFF1F2;border:1px solid #FECDD3;
                        border-radius:8px;padding:10px;text-align:center;'>
                <div style='font-size:18px;font-weight:700;color:#E11D48;'>
                    {n_pos}</div>
                <div style='font-size:10px;color:#BE123C;font-weight:600;'>
                    Positive</div>
            </div>
            <div style='background:#F0FDF4;border:1px solid #BBF7D0;
                        border-radius:8px;padding:10px;text-align:center;'>
                <div style='font-size:18px;font-weight:700;color:#059669;'>
                    {len(hist)-n_pos}</div>
                <div style='font-size:10px;color:#047857;font-weight:600;'>
                    Negative</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#E8EBF0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#F9FAFB;border:1px solid #E8EBF0;border-radius:10px;
                padding:12px 14px;text-align:center;'>
        <p style='color:#9CA3AF; font-size:11px; margin:0; line-height:1.7;'>
            ⚕️ <b style='color:#6B7280;'>Educational use only</b><br>
            Not a substitute for clinical diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# MAIN HEADER
# ════════════════════════════════════════════════════════
st.markdown("""
<div style='padding:32px 0 20px;'>
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:10px;'>
        <div style='width:44px;height:44px;background:linear-gradient(135deg,#6366F1,#8B5CF6);
                    border-radius:12px;display:inline-flex;align-items:center;
                    justify-content:center;font-size:22px;
                    box-shadow:0 4px 12px rgba(99,102,241,0.3);'>🧠</div>
        <div>
            <h1 style='font-family:Playfair Display,serif; font-size:32px;
                       font-weight:600; color:#111827; margin:0; line-height:1.2;'>
                Alzheimer's Disease
                <span style='color:#6366F1;'>Prediction System</span>
            </h1>
        </div>
    </div>
    <p style='color:#6B7280; font-size:14px; margin:0; padding-left:58px;'>
        Single patient analysis &nbsp;·&nbsp; Batch CSV/Excel upload
        &nbsp;·&nbsp; Analytics dashboard &nbsp;·&nbsp;
        History &nbsp;·&nbsp; Power BI export
    </p>
</div>
<hr style='border-color:#E8EBF0; margin-bottom:0;'>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════
tab_single, tab_batch, tab_dashboard, tab_history, tab_download = st.tabs([
    "🔍 Single Predict",
    "📂 Batch Upload",
    "📊 Dashboard",
    "🕓 History",
    "⬇️ Download & Power BI"
])


# ════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICT
# ════════════════════════════════════════════════════════
with tab_single:
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("""
        <h3 style='font-family:Playfair Display,serif; color:#111827;
                   font-weight:600; font-size:18px; margin-bottom:20px;'>
            👤 Patient Assessment
        </h3>""", unsafe_allow_html=True)

        patient_id = st.text_input("Patient ID (optional)",
                                   placeholder="e.g. PAT-00123")

        st.markdown("""
        <p style='font-size:11px;font-weight:600;color:#6B7280;
                  text-transform:uppercase;letter-spacing:0.08em;
                  margin:16px 0 6px;'>🧪 Cognitive Tests</p>
        """, unsafe_allow_html=True)

        mmse_val = st.slider("MMSE Score (Mini Mental State Exam)",
                             0.0, 30.0, 25.0, 0.5,
                             help="24-30 Normal | 18-23 Mild | 0-17 Severe")
        mmse_status = ("🟢 Normal (≥24)"   if mmse_val >= 24 else
                       "🟡 Mild (18–23)"   if mmse_val >= 18 else
                       "🔴 Severe (<18)")
        mmse_color  = ("#059669" if mmse_val >= 24 else
                       "#D97706" if mmse_val >= 18 else "#DC2626")
        st.markdown(f"""
        <p style='color:{mmse_color}; font-size:12px; font-weight:600;
                  margin:-8px 0 16px;'>{mmse_status}</p>
        """, unsafe_allow_html=True)

        func_val = st.slider("Functional Assessment",
                             0.0, 10.0, 7.0, 0.1,
                             help="0=Severe impairment | 10=Fully functional")
        adl_val  = st.slider("ADL Score (Activities of Daily Living)",
                             0.0, 10.0, 7.0, 0.1,
                             help="0=Cannot perform | 10=Independent")

        st.markdown("""
        <p style='font-size:11px;font-weight:600;color:#6B7280;
                  text-transform:uppercase;letter-spacing:0.08em;
                  margin:16px 0 6px;'>🧩 Behavioral Symptoms</p>
        """, unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            mem_sel = st.selectbox("Memory Complaints",
                                   ["No  ✅", "Yes  ⚠️"])
            mem_val = 1 if "Yes" in mem_sel else 0
        with s2:
            beh_sel = st.selectbox("Behavioral Problems",
                                   ["No  ✅", "Yes  ⚠️"])
            beh_val = 1 if "Yes" in beh_sel else 0

        # Risk flag badges
        flags = []
        if mmse_val  < 24: flags.append("Low MMSE")
        if func_val  <  5: flags.append("Low Functional")
        if adl_val   <  5: flags.append("Low ADL")
        if mem_val  == 1:  flags.append("Memory Complaints")
        if beh_val  == 1:  flags.append("Behavioral Problems")
        if flags:
            badges = " &nbsp;".join([
                f"<span style='background:#FFF1F2;"
                f"border:1px solid #FECDD3;"
                f"border-radius:6px; padding:3px 10px;"
                f"font-size:11px; font-weight:600; color:#E11D48;'>"
                f"⚑ {f}</span>"
                for f in flags])
            st.markdown(f"""
            <div style='margin:12px 0 16px; padding:10px 14px;
                        background:#FFF1F2; border:1px solid #FECDD3;
                        border-radius:10px;'>
                <p style='color:#9F1239; font-size:11px; font-weight:600;
                          margin:0 0 6px; text-transform:uppercase;
                          letter-spacing:0.05em;'>
                    Risk factors detected
                </p>
                {badges}
            </div>""", unsafe_allow_html=True)

        predict_btn = st.button("🔍  Run Prediction",
                                use_container_width=True)

    # ── RIGHT: Results ──────────────────────────────────
    with right:
        st.markdown("""
        <h3 style='font-family:Playfair Display,serif; color:#111827;
                   font-weight:600; font-size:18px; margin-bottom:20px;'>
            📋 Prediction Result
        </h3>""", unsafe_allow_html=True)

        if predict_btn:
            vals = {
                'FunctionalAssessment': func_val,
                'MemoryComplaints'    : mem_val,
                'ADL'                 : adl_val,
                'BehavioralProblems'  : beh_val,
                'MMSE'                : mmse_val,
            }
            prob, pred = predict_single(vals)
            prob_pct   = prob * 100
            rlabel, rcolor = risk_label(prob)

            if pred == 1:
                card_cls = "card-ad"
                icon     = "⚠️"
                headline = "Alzheimer's Risk Detected"
                subtext  = ("Elevated risk detected. "
                            "Immediate clinical evaluation recommended.")
                txt_c    = "#991B1B"
                sub_c    = "#B91C1C"
            else:
                card_cls = "card-healthy"
                icon     = "✅"
                headline = "No Alzheimer's Detected"
                subtext  = ("Low risk predicted. "
                            "Continue regular monitoring.")
                txt_c    = "#065F46"
                sub_c    = "#047857"

            st.markdown(f"""
            <div class='result-card {card_cls}'>
                <div style='font-size:48px; margin-bottom:10px;'>{icon}</div>
                <h2 style='font-family:Playfair Display,serif; color:{txt_c};
                           font-size:22px; font-weight:600; margin:0 0 10px;'>
                    {headline}
                </h2>
                <p style='color:{sub_c}; font-size:13px;
                          margin:0 0 16px; line-height:1.6;'>{subtext}</p>
                <div style='display:inline-block; background:white;
                            border:1.5px solid {rcolor}; border-radius:8px;
                            padding:6px 18px; font-size:13px; font-weight:700;
                            color:{rcolor};'>
                    {rlabel}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability display
            prob_txt_c = "#DC2626" if pred==1 else "#059669"
            st.markdown(f"""
            <div style='text-align:center; margin:20px 0 8px;
                        background:#FFFFFF; border:1px solid #E8EBF0;
                        border-radius:14px; padding:20px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
                <p style='color:#9CA3AF; font-size:11px; font-weight:600;
                          text-transform:uppercase; letter-spacing:0.08em;
                          margin:0;'>Alzheimer's Probability</p>
                <p style='font-size:52px; font-weight:800; color:{prob_txt_c};
                          margin:6px 0 4px; font-family:Playfair Display,serif;
                          line-height:1;'>
                    {prob_pct:.1f}%
                </p>
                <p style='color:#9CA3AF; font-size:12px; margin:0;'>
                    Threshold: {threshold} &nbsp;·&nbsp;
                    {"Above → Positive" if pred==1 else "Below → Negative"}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Probability bar chart
            fig_p, ax_p = dark_fig(5, 0.8)
            ax_p.barh([''], [1.0], color='#E8EBF0', height=0.55)
            ax_p.barh([''], [prob], color=rcolor, height=0.55, alpha=0.9)
            ax_p.axvline(threshold, color='#6366F1', lw=1.5,
                         ls='--', alpha=0.8)
            ax_p.set_xlim(0, 1)
            ax_p.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_p.set_xticklabels(['0%','25%','50%','75%','100%'],
                                  color='#6B7280', fontsize=9)
            ax_p.set_yticks([])
            for s in ax_p.spines.values():
                s.set_visible(False)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

            # Feature values vs healthy range
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:16px 0 8px;
                      text-transform:uppercase; letter-spacing:1px;'>
                📊 Feature Values vs Healthy Range
            </p>""", unsafe_allow_html=True)

            patient_vals_map = {
                'MMSE': mmse_val, 'FunctionalAssessment': func_val,
                'ADL': adl_val, 'MemoryComplaints': mem_val,
                'BehavioralProblems': beh_val,
            }
            maxv = [MAX_VALS[f] for f in features]
            p_n  = [patient_vals_map[f]/m for f,m in zip(features, maxv)]
            hr_n = [(HEALTHY_RANGES[f][0]/m,
                     HEALTHY_RANGES[f][1]/m)
                    for f,m in zip(features, maxv)]
            bcols = []
            for pn, (lo, hi) in zip(p_n, hr_n):
                if lo <= pn <= hi: bcols.append('#10b981')
                elif abs(pn-lo)<0.15 or abs(pn-hi)<0.15:
                    bcols.append('#f59e0b')
                else: bcols.append('#f43f5e')

            fig_f, ax_f = dark_fig(5, 3.2)
            xlabs = ['ADL','Func.\nAssess.','MMSE\n(/30)',
                     'Memory\nCompl.','Behavioral\nProb.']
            xpos  = np.arange(len(features))
            ax_f.bar(xpos, p_n, color=bcols,
                     alpha=0.85, width=0.55, edgecolor='none')
            for i,(lo,hi) in enumerate(hr_n):
                ax_f.hlines([lo,hi], i-.3, i+.3,
                             colors='#9CA3AF', lw=1.5,
                             linestyles='--', alpha=0.7)
            raw = [adl_val, func_val, mmse_val, mem_val, beh_val]
            for i,(v,pn) in enumerate(zip(raw, p_n)):
                ax_f.text(i, pn+0.04,
                          f'{v:.1f}' if v>1 else str(int(v)),
                          ha='center', color='#374151',
                          fontsize=9, fontweight='600')
            ax_f.set_xticks(xpos)
            ax_f.set_xticklabels(xlabs, color='#374151', fontsize=8)
            ax_f.set_ylim(0, 1.35)
            ax_f.set_yticks([0,.25,.5,.75,1.0])
            ax_f.set_yticklabels(['0%','25%','50%','75%','100%'],
                                   color='#6B7280', fontsize=8)
            ax_f.tick_params(length=0)
            for s in ax_f.spines.values():
                s.set_visible(False)
            ax_f.grid(True, axis='y', alpha=0.5, color='#E5E7EB', lw=0.8)
            ax_f.legend(handles=[
                mpatches.Patch(color='#10b981', label='Normal'),
                mpatches.Patch(color='#f59e0b', label='Borderline'),
                mpatches.Patch(color='#f43f5e', label='Abnormal'),
            ], loc='upper right', fontsize=7,
               facecolor='#FFFFFF', labelcolor='#374151',
               framealpha=0.8, edgecolor='#E8EBF0')
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_f, use_container_width=True)
            plt.close()

            # Save to history
            append_history({
                "timestamp"             : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_id"            : patient_id or f"PAT-{datetime.now().strftime('%H%M%S')}",
                "adl"                   : adl_val,
                "functional_assessment" : func_val,
                "mmse"                  : mmse_val,
                "memory_complaints"     : mem_val,
                "behavioral_problems"   : beh_val,
                "probability"           : round(prob, 4),
                "risk_level"            : rlabel,
                "prediction"            : "Alzheimer's Detected" if pred==1
                                          else "No Alzheimer's",
            })
            st.success("✅ Result saved to history automatically.")

            # Single result download
            single_df = pd.DataFrame([{
                "Patient ID"             : patient_id or "—",
                "MMSE"                   : mmse_val,
                "Functional Assessment"  : func_val,
                "ADL"                    : adl_val,
                "Memory Complaints"      : "Yes" if mem_val else "No",
                "Behavioral Problems"    : "Yes" if beh_val else "No",
                "Probability (%)"        : round(prob_pct, 2),
                "Risk Level"             : rlabel,
                "Prediction"             : "Alzheimer's Detected"
                                           if pred==1 else "No Alzheimer's",
                "Timestamp"              : datetime.now().strftime(
                                           "%Y-%m-%d %H:%M:%S"),
            }])
            st.download_button(
                "⬇️  Download this result (.xlsx)",
                data=to_excel_bytes(single_df, "SingleResult"),
                file_name=f"result_{patient_id or 'patient'}.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True,
            )

        else:
            # Placeholder + feature importance
            st.markdown("""
            <div style='background:#FFFFFF; border:2px dashed #D1D5DB;
                        border-radius:16px; padding:50px 32px;
                        text-align:center; margin-top:12px;'>
                <div style='font-size:52px; margin-bottom:12px;
                            opacity:0.35;'>🧠</div>
                <h3 style='font-family:Playfair Display,serif; color:#374151;
                           font-weight:400; font-size:20px; margin:0 0 8px;'>
                    Awaiting Patient Data
                </h3>
                <p style='color:#374151; font-size:14px;
                          line-height:1.6; margin:0;'>
                    Fill in assessment scores on the left<br>
                    and click <b style='color:#6366F1;'>Run Prediction</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:20px 0 8px;
                      text-transform:uppercase; letter-spacing:1px;'>
                🏆 Feature Importance (Model Weights)
            </p>""", unsafe_allow_html=True)

            imp_items = sorted(feat_imp.items(),
                               key=lambda x: x[1], reverse=True)
            fig_i, ax_i = dark_fig(5, 3)
            names = [k.replace('Assessment','Assess.')
                      .replace('Complaints','Compl.')
                      .replace('Problems','Prob.')
                     for k,_ in imp_items]
            vals2 = [v for _,v in imp_items]
            cols2 = ['#6366F1','#7C3AED','#8B5CF6','#A78BFA','#C4B5FD']
            bars  = ax_i.barh(names, vals2,
                               color=cols2[:len(names)],
                               alpha=0.9, height=0.55)
            for b,v in zip(bars, vals2):
                ax_i.text(v+0.003, b.get_y()+b.get_height()/2,
                          f'{v:.3f}', va='center',
                          color='#374151', fontsize=9, fontweight='600')
            ax_i.invert_yaxis()
            ax_i.set_xlim(0, max(vals2)*1.35)
            ax_i.tick_params(colors='#374151', labelsize=9, length=0)
            for s in ax_i.spines.values():
                s.set_visible(False)
            ax_i.grid(True, axis='x', alpha=0.5,
                       color='#E5E7EB', lw=0.8)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_i, use_container_width=True)
            plt.close()


# ════════════════════════════════════════════════════════
# TAB 2 — BATCH UPLOAD
# ════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<br>", unsafe_allow_html=True)

    # Template download
    st.markdown("""
    <h3 style='font-family:Playfair Display,serif; color:#111827;
               font-weight:400; margin-bottom:4px;'>
        📂 Batch Prediction
    </h3>
    <p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>
        Upload a CSV or Excel file with multiple patients.
        The app will predict all at once and generate a dashboard.
    </p>
    """, unsafe_allow_html=True)

    # Template
    template_df = pd.DataFrame(columns=[
        'patient_id', 'ADL', 'FunctionalAssessment',
        'MMSE', 'MemoryComplaints', 'BehavioralProblems'
    ])
    sample_rows = [
        ['PAT-001', 8.2, 7.5, 26.0, 0, 0],
        ['PAT-002', 4.1, 3.8, 18.0, 1, 1],
        ['PAT-003', 6.0, 5.5, 22.0, 1, 0],
    ]
    for r in sample_rows:
        template_df.loc[len(template_df)] = r

    col_t, col_s = st.columns([2,1])
    with col_t:
        st.download_button(
            "📄  Download upload template (.xlsx)",
            data=to_excel_bytes(template_df, "Template"),
            file_name="neuroscan_upload_template.xlsx",
            mime=("application/vnd.openxmlformats-officedocument"
                  ".spreadsheetml.sheet"),
            use_container_width=True,
        )
    with col_s:
        st.markdown("""
        <div class='info-box' style='padding:10px 14px;'>
            <p style='margin:0; color:#6B7280; font-size:12px;'>
                Required columns:<br>
                <code style='color:#6366F1;'>ADL · FunctionalAssessment
                · MMSE · MemoryComplaints · BehavioralProblems</code>
            </p>
        </div>""", unsafe_allow_html=True)

    # Upload
    uploaded = st.file_uploader(
        "Upload patient file (.xlsx or .csv)",
        type=['xlsx', 'csv']
    )

    if uploaded is not None:
        raw_df = (pd.read_excel(uploaded)
                  if uploaded.name.endswith('.xlsx')
                  else pd.read_csv(uploaded))

        st.markdown(f"""
        <div class='info-box'>
            <p style='margin:0; color:#6B7280; font-size:13px;'>
                ✅ File loaded: <b style='color:#111827;'>
                {uploaded.name}</b> &nbsp;·&nbsp;
                <b style='color:#6366F1;'>{len(raw_df)} patients</b>
                detected
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Preview raw data (first 5 rows)"):
            st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("🚀  Run Batch Predictions",
                     use_container_width=True):
            with st.spinner(f"Running model on {len(raw_df)} patients..."):
                try:
                    results_df = predict_batch(raw_df.copy())
                    st.session_state['batch_results'] = results_df
                except KeyError as e:
                    st.error(f"Missing column: {e}. "
                             "Please use the template above.")
                    st.stop()

    # Show results if available
    if 'batch_results' in st.session_state:
        res = st.session_state['batch_results']

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='font-family:Playfair Display,serif; color:#111827;
                   font-weight:400; margin:16px 0 12px;'>
            📊 Batch Results
        </h3>""", unsafe_allow_html=True)

        # Summary cards
        total   = len(res)
        n_high  = len(res[res.risk_level == "High Risk"])
        n_med   = len(res[res.risk_level == "Medium Risk"])
        n_low   = len(res[res.risk_level == "Low Risk"])
        n_pos   = len(res[res.prediction.str.contains("Detected", na=False)])
        avg_p   = res.probability.mean() * 100

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Total", total)
        c2.metric("Positive", n_pos)
        c3.metric("High Risk",   n_high)
        c4.metric("Medium Risk", n_med)
        c5.metric("Low Risk",    n_low)
        c6.metric("Avg Prob.",   f"{avg_p:.1f}%")

        # Mini charts
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:12px 0 6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                Risk distribution
            </p>""", unsafe_allow_html=True)
            fig_d, ax_d = dark_fig(4, 2.6)
            sizes  = [n_high, n_med, n_low]
            colors = ['#f43f5e', '#f59e0b', '#10b981']
            labels = [f'High\n{n_high}',
                      f'Medium\n{n_med}',
                      f'Low\n{n_low}']
            wedges, _ = ax_d.pie(
                sizes, colors=colors,
                wedgeprops=dict(width=0.55, edgecolor='#FFFFFF'),
                startangle=90
            )
            ax_d.legend(wedges, labels,
                        loc="center left", bbox_to_anchor=(1,0.5),
                        fontsize=9, facecolor='#FFFFFF',
                        labelcolor='#374151',
                        edgecolor='#E8EBF0')
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close()

        with ch2:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:12px 0 6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                Probability distribution
            </p>""", unsafe_allow_html=True)
            fig_h, ax_h = dark_fig(4, 2.6)
            ax_h.hist(res.probability * 100, bins=15,
                      color='#6366F1', alpha=0.8,
                      edgecolor='#FFFFFF')
            ax_h.axvline(threshold*100, color='#f59e0b',
                          lw=1.5, ls='--', alpha=0.8,
                          label=f'Threshold {threshold}')
            ax_h.tick_params(colors='#6B7280', labelsize=8, length=0)
            ax_h.set_xlabel('Probability (%)',
                             color='#6B7280', fontsize=9)
            ax_h.set_ylabel('Count',
                             color='#6B7280', fontsize=9)
            for s in ax_h.spines.values():
                s.set_visible(False)
            ax_h.grid(True, axis='y', alpha=0.08,
                       color='#E5E7EB', lw=0.8)
            ax_h.legend(fontsize=8, facecolor='#FFFFFF',
                         labelcolor='#374151',
                         edgecolor='#E8EBF0')
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_h, use_container_width=True)
            plt.close()

        # Results table sorted by probability
        st.markdown("""
        <p style='color:#6B7280; font-size:11px; font-weight:600; margin:12px 0 6px;
                  text-transform:uppercase; letter-spacing:1px;'>
            All results — sorted by risk (highest first)
        </p>""", unsafe_allow_html=True)

        display_cols = ([c for c in ['patient_id'] if c in res.columns] +
                        ['ADL', 'FunctionalAssessment', 'MMSE',
                         'MemoryComplaints', 'BehavioralProblems',
                         'probability', 'risk_level', 'prediction'])
        display_cols = [c for c in display_cols if c in res.columns]

        st.dataframe(
            res[display_cols]
              .sort_values('probability', ascending=False)
              .reset_index(drop=True),
            use_container_width=True
        )

        # Download buttons
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇️  Download all results (.xlsx)",
                data=to_excel_bytes(res, "BatchResults"),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️  Download high-risk only (.xlsx)",
                data=to_excel_bytes(
                    res[res.risk_level == "High Risk"], "HighRisk"),
                file_name="high_risk_patients.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True,
            )

        # Append to history
        hist_rows = []
        for _, row in res.iterrows():
            hist_rows.append({
                "timestamp"             : row.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "patient_id"            : row.get('patient_id', '—'),
                "adl"                   : row.get('ADL', ''),
                "functional_assessment" : row.get('FunctionalAssessment', ''),
                "mmse"                  : row.get('MMSE', ''),
                "memory_complaints"     : row.get('MemoryComplaints', ''),
                "behavioral_problems"   : row.get('BehavioralProblems', ''),
                "probability"           : row.get('probability', ''),
                "risk_level"            : row.get('risk_level', ''),
                "prediction"            : row.get('prediction', ''),
            })
        hist_df = load_history()
        hist_df = pd.concat([hist_df,
                              pd.DataFrame(hist_rows)],
                             ignore_index=True)
        hist_df.to_excel(HISTORY_FILE, index=False)
        st.success(f"✅ {len(res)} records saved to history.")


# ════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("<br>", unsafe_allow_html=True)
    hist = load_history()

    if hist.empty:
        st.markdown("""
        <div style='background:#FFFFFF; border:2px dashed #D1D5DB;
                    border-radius:16px; padding:60px 32px;
                    text-align:center;'>
            <div style='font-size:48px; opacity:0.3; margin-bottom:12px;'>
                📊</div>
            <h3 style='font-family:Playfair Display,serif; color:#374151;
                       font-weight:400;'>No data yet</h3>
            <p style='color:#374151;'>
                Run single or batch predictions to populate the dashboard.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        hist['timestamp'] = pd.to_datetime(
            hist['timestamp'], errors='coerce')
        hist['date'] = hist['timestamp'].dt.date

        # KPI row
        st.markdown("""
        <h3 style='font-family:Playfair Display,serif; color:#111827;
                   font-weight:400; margin-bottom:16px;'>
            📊 Analytics Dashboard
        </h3>""", unsafe_allow_html=True)

        total  = len(hist)
        n_pos  = hist.prediction.str.contains(
            "Detected", na=False).sum()
        n_neg  = total - n_pos
        avg_p  = hist.probability.mean() * 100
        hi_rt  = hist.risk_level.str.contains(
            "High", na=False).mean() * 100

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Predictions", total)
        k2.metric("Positive (AD)", n_pos,
                  delta=f"{n_pos/total*100:.0f}%")
        k3.metric("Avg Probability", f"{avg_p:.1f}%")
        k4.metric("High Risk Rate",  f"{hi_rt:.1f}%")

        st.markdown("<hr>", unsafe_allow_html=True)

        row1_l, row1_r = st.columns(2)

        # Trend over time
        with row1_l:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin-bottom:6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                Daily avg probability trend
            </p>""", unsafe_allow_html=True)
            daily = (hist.dropna(subset=['date'])
                        .groupby('date')['probability']
                        .mean().reset_index())
            if len(daily) > 1:
                fig_t, ax_t = dark_fig(5, 2.8)
                ax_t.plot(range(len(daily)),
                          daily.probability * 100,
                          color='#6366F1', lw=2, marker='o',
                          markersize=4)
                ax_t.fill_between(
                    range(len(daily)),
                    daily.probability * 100,
                    alpha=0.15, color='#6366F1')
                ax_t.axhline(threshold*100, color='#f59e0b',
                              lw=1, ls='--', alpha=0.6)
                if len(daily) <= 10:
                    ax_t.set_xticks(range(len(daily)))
                    ax_t.set_xticklabels(
                        [str(d) for d in daily.date],
                        color='#6B7280', fontsize=7, rotation=30)
                ax_t.set_ylabel('Avg probability (%)',
                                 color='#6B7280', fontsize=8)
                style_ax(ax_t)
                plt.tight_layout(pad=0.4)
                st.pyplot(fig_t, use_container_width=True)
                plt.close()
            else:
                st.info("Need more data points for trend chart.")

        # Risk breakdown donut
        with row1_r:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin-bottom:6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                Risk level breakdown
            </p>""", unsafe_allow_html=True)
            rc = hist.risk_level.value_counts()
            fig_pie, ax_pie = dark_fig(5, 2.8)
            pie_cols = {'High Risk':'#f43f5e',
                        'Medium Risk':'#f59e0b',
                        'Low Risk':'#10b981'}
            clrs = [pie_cols.get(l, '#9CA3AF') for l in rc.index]
            weds, txts, autotxts = ax_pie.pie(
                rc.values, labels=rc.index,
                colors=clrs, autopct='%1.0f%%',
                wedgeprops=dict(width=0.55, edgecolor='#FFFFFF'),
                startangle=90, pctdistance=0.75
            )
            for t in txts:
                t.set_color('#374151')
                t.set_fontsize(9)
            for at in autotxts:
                at.set_color('white')
                at.set_fontsize(8)
            plt.tight_layout()
            st.pyplot(fig_pie, use_container_width=True)
            plt.close()

        row2_l, row2_r = st.columns(2)

        # MMSE distribution by outcome
        with row2_l:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:12px 0 6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                MMSE score distribution by outcome
            </p>""", unsafe_allow_html=True)
            pos_mmse = hist[
                hist.prediction.str.contains("Detected", na=False)
            ]['mmse'].dropna()
            neg_mmse = hist[
                ~hist.prediction.str.contains("Detected", na=False)
            ]['mmse'].dropna()
            fig_m, ax_m = dark_fig(5, 2.8)
            if len(pos_mmse):
                ax_m.hist(pos_mmse, bins=10, alpha=0.7,
                           color='#f43f5e', label="AD Detected",
                           edgecolor='#FFFFFF')
            if len(neg_mmse):
                ax_m.hist(neg_mmse, bins=10, alpha=0.7,
                           color='#10b981', label="No AD",
                           edgecolor='#FFFFFF')
            ax_m.set_xlabel('MMSE Score',
                             color='#6B7280', fontsize=9)
            ax_m.set_ylabel('Count',
                             color='#6B7280', fontsize=9)
            ax_m.tick_params(colors='#6B7280', labelsize=8, length=0)
            for s in ax_m.spines.values():
                s.set_visible(False)
            ax_m.grid(True, axis='y', alpha=0.08,
                       color='#E5E7EB', lw=0.8)
            ax_m.legend(fontsize=8, facecolor='#FFFFFF',
                         labelcolor='#374151',
                         edgecolor='#E8EBF0')
            plt.tight_layout(pad=0.4)
            st.pyplot(fig_m, use_container_width=True)
            plt.close()

        # Avg feature values by risk
        with row2_r:
            st.markdown("""
            <p style='color:#6B7280; font-size:11px; font-weight:600; margin:12px 0 6px;
                      text-transform:uppercase; letter-spacing:1px;'>
                Avg feature values by risk level
            </p>""", unsafe_allow_html=True)
            feat_cols = ['adl','functional_assessment',
                         'mmse','memory_complaints',
                         'behavioral_problems']
            avail = [c for c in feat_cols if c in hist.columns]
            if avail and 'risk_level' in hist.columns:
                grp = hist.groupby('risk_level')[avail].mean()
                fig_g, ax_g = dark_fig(5, 2.8)
                x  = np.arange(len(avail))
                bw = 0.25
                rlevels = ['Low Risk','Medium Risk','High Risk']
                rcolors = ['#10b981','#f59e0b','#f43f5e']
                for i,(rl,rc2) in enumerate(zip(rlevels,rcolors)):
                    if rl in grp.index:
                        norms = [grp.loc[rl,c] /
                                  MAX_VALS.get(c.replace('_','').title(), 10)
                                  for c in avail]
                        ax_g.bar(x + i*bw, norms, bw,
                                  color=rc2, alpha=0.8,
                                  label=rl, edgecolor='#FFFFFF')
                short = ['ADL','Func.','MMSE',
                         'Memory','Behav.']
                ax_g.set_xticks(x + bw)
                ax_g.set_xticklabels(
                    short[:len(avail)],
                    color='#374151', fontsize=8)
                ax_g.set_ylabel('Normalised value',
                                 color='#6B7280', fontsize=8)
                ax_g.tick_params(colors='#6B7280',
                                  labelsize=8, length=0)
                for s in ax_g.spines.values():
                    s.set_visible(False)
                ax_g.grid(True, axis='y', alpha=0.08,
                           color='#E5E7EB', lw=0.8)
                ax_g.legend(fontsize=7, facecolor='#FFFFFF',
                             labelcolor='#374151',
                             edgecolor='#E8EBF0')
                plt.tight_layout(pad=0.4)
                st.pyplot(fig_g, use_container_width=True)
                plt.close()

        # Recent predictions table
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#6B7280; font-size:11px; font-weight:600; margin-bottom:8px;
                  text-transform:uppercase; letter-spacing:1px;'>
            Recent predictions
        </p>""", unsafe_allow_html=True)
        st.dataframe(
            hist.sort_values('timestamp', ascending=False)
                .head(10)
                .reset_index(drop=True),
            use_container_width=True
        )


# ════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ════════════════════════════════════════════════════════
with tab_history:
    st.markdown("<br>", unsafe_allow_html=True)
    hist = load_history()

    st.markdown("""
    <h3 style='font-family:Playfair Display,serif; color:#111827;
               font-weight:400; margin-bottom:16px;'>
        🕓 Prediction History
    </h3>""", unsafe_allow_html=True)

    if hist.empty:
        st.info("No predictions yet. "
                "Run single or batch predictions to build history.")
    else:
        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            search = st.text_input("Search patient ID", "")
        with f2:
            risk_f = st.selectbox(
                "Risk level",
                ["All","High Risk","Medium Risk","Low Risk"])
        with f3:
            pred_f = st.selectbox(
                "Prediction",
                ["All","Alzheimer's Detected","No Alzheimer's"])

        filtered = hist.copy()
        if search:
            filtered = filtered[
                filtered.patient_id.astype(str)
                         .str.contains(search, case=False, na=False)]
        if risk_f != "All":
            filtered = filtered[
                filtered.risk_level == risk_f]
        if pred_f != "All":
            filtered = filtered[
                filtered.prediction.str.contains(
                    pred_f.split("'")[0], na=False)]

        # Quick stats
        h1,h2,h3,h4 = st.columns(4)
        h1.metric("Showing", len(filtered))
        h2.metric("Total",   len(hist))
        h3.metric("High Risk",
                  len(filtered[filtered.risk_level
                               .str.contains("High", na=False)]))
        h4.metric("Avg Prob.",
                  f"{filtered.probability.mean()*100:.1f}%"
                  if not filtered.empty else "—")

        st.dataframe(
            filtered.sort_values('timestamp', ascending=False)
                    .reset_index(drop=True),
            use_container_width=True
        )

        # Clear history
        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("🗑️  Clear all history", use_container_width=False):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared.")
            st.rerun()


# ════════════════════════════════════════════════════════
# TAB 5 — DOWNLOAD & POWER BI
# ════════════════════════════════════════════════════════
with tab_download:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h3 style='font-family:Playfair Display,serif; color:#111827;
               font-weight:400; margin-bottom:4px;'>
        ⬇️ Download & Export
    </h3>
    <p style='color:#6B7280; font-size:14px; margin-bottom:20px;'>
        Export your prediction history to Excel or CSV.
        Use the Power BI section to connect your dashboard.
    </p>
    """, unsafe_allow_html=True)

    hist = load_history()

    # ── Export filters ────────────────────────────────────
    st.markdown("""
    <div class='info-box'>
        <p style='margin:0 0 12px; color:#6B7280; font-size:13px;
                  font-weight:600; text-transform:uppercase;
                  letter-spacing:1px;'>Export Filters</p>
    """, unsafe_allow_html=True)

    ef1, ef2 = st.columns(2)
    with ef1:
        risk_dl = st.selectbox(
            "Risk level filter",
            ["All","High Risk","Medium Risk","Low Risk"],
            key="dl_risk")
    with ef2:
        pred_dl = st.selectbox(
            "Prediction filter",
            ["All","Alzheimer's Detected","No Alzheimer's"],
            key="dl_pred")

    st.markdown("</div>", unsafe_allow_html=True)

    export_df = hist.copy()
    if risk_dl != "All":
        export_df = export_df[export_df.risk_level == risk_dl]
    if pred_dl != "All":
        export_df = export_df[
            export_df.prediction.str.contains(
                pred_dl.split("'")[0], na=False)]

    # Summary
    st.markdown(f"""
    <div class='info-box' style='margin:12px 0;'>
        <div style='display:flex; gap:32px; flex-wrap:wrap;'>
            <div>
                <p style='margin:0; color:#6B7280; font-size:11px;
                          text-transform:uppercase; letter-spacing:1px;'>
                    Records to export</p>
                <p style='margin:4px 0 0; font-size:22px;
                          font-weight:600; color:#111827;'>
                    {len(export_df)}</p>
            </div>
            <div>
                <p style='margin:0; color:#6B7280; font-size:11px;
                          text-transform:uppercase; letter-spacing:1px;'>
                    High risk</p>
                <p style='margin:4px 0 0; font-size:22px;
                          font-weight:600; color:#f43f5e;'>
                    {len(export_df[export_df.risk_level.str.contains("High",na=False)]) if not export_df.empty else 0}
                </p>
            </div>
            <div>
                <p style='margin:0; color:#6B7280; font-size:11px;
                          text-transform:uppercase; letter-spacing:1px;'>
                    Avg probability</p>
                <p style='margin:4px 0 0; font-size:22px;
                          font-weight:600; color:#6366F1;'>
                    {f"{export_df.probability.mean()*100:.1f}%" if not export_df.empty else "—"}
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Download buttons
    b1, b2 = st.columns(2)
    fname  = f"neuroscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with b1:
        if not export_df.empty:
            st.download_button(
                "⬇️  Download Excel (.xlsx)",
                data=to_excel_bytes(export_df, "PredictionHistory"),
                file_name=f"{fname}.xlsx",
                mime=("application/vnd.openxmlformats-officedocument"
                      ".spreadsheetml.sheet"),
                use_container_width=True,
            )
        else:
            st.button("⬇️  Download Excel (.xlsx)",
                      disabled=True, use_container_width=True)

    with b2:
        if not export_df.empty:
            st.download_button(
                "⬇️  Download CSV (.csv)",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{fname}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️  Download CSV (.csv)",
                      disabled=True, use_container_width=True)

    # ── Power BI section ──────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <h3 style='font-family:Playfair Display,serif; color:#111827;
               font-weight:400; margin-bottom:4px;'>
        📊 Power BI Dashboard Integration
    </h3>
    <p style='color:#6B7280; font-size:14px; margin-bottom:16px;'>
        Connect your Excel history file to Power BI for
        executive-level reporting and trend analysis.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='powerbi-box'>
        <p style='color:#f3a11a; font-size:13px; font-weight:600;
                  margin:0 0 12px; text-transform:uppercase;
                  letter-spacing:1px;'>
            ⚡ How to connect Power BI
        </p>
        <div style='color:#6B7280; font-size:13px; line-height:2.2;'>
            <b style='color:#111827;'>Step 1</b> &nbsp;—&nbsp;
            Download the Excel file above
            (<code style='color:#f3a11a;'>neuroscan_YYYYMMDD.xlsx</code>)
            <br>
            <b style='color:#111827;'>Step 2</b> &nbsp;—&nbsp;
            Open Power BI Desktop → Get Data → Excel Workbook
            <br>
            <b style='color:#111827;'>Step 3</b> &nbsp;—&nbsp;
            Select the
            <code style='color:#f3a11a;'>PredictionHistory</code>
            sheet → Load
            <br>
            <b style='color:#111827;'>Step 4</b> &nbsp;—&nbsp;
            Build visuals: Risk donut · Trend line ·
            MMSE distribution · Patient table
            <br>
            <b style='color:#111827;'>Step 5</b> &nbsp;—&nbsp;
            To refresh: replace the Excel file and click
            <b style='color:#f3a11a;'>Refresh</b> in Power BI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Recommended Power BI columns reference
    st.markdown("""
    <p style='color:#6B7280; font-size:11px; font-weight:600; margin:16px 0 8px;
              text-transform:uppercase; letter-spacing:1px;'>
        Exported column reference for Power BI
    </p>""", unsafe_allow_html=True)

    ref = pd.DataFrame({
        "Column"       : ["timestamp","patient_id","mmse",
                           "adl","functional_assessment",
                           "memory_complaints","behavioral_problems",
                           "probability","risk_level","prediction"],
        "Power BI type": ["DateTime","Text","Decimal",
                           "Decimal","Decimal",
                           "Whole Number","Whole Number",
                           "Decimal","Text","Text"],
        "Use for"      : ["Trend axis","Filter/slicer",
                           "Distribution chart",
                           "Avg by risk group",
                           "Avg by risk group",
                           "Count flag","Count flag",
                           "KPI gauge","Donut / slicer",
                           "Bar chart / count"],
    })
    st.dataframe(ref, hide_index=True, use_container_width=True)

    # Auto-refresh tip
    st.markdown("""
    <div class='info-box' style='margin-top:12px;'>
        <p style='margin:0; color:#6B7280; font-size:13px;
                  line-height:1.8;'>
            <b style='color:#6B7280;'>💡 Pro tip:</b>
            For live auto-refresh in Power BI Service, save the Excel
            file to a <b>SharePoint / OneDrive</b> folder and connect
            Power BI to that cloud path. Every time the Streamlit app
            writes a new prediction, Power BI will pick it up on the
            next scheduled refresh (set to 30 min or hourly).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature reference
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#6B7280; font-size:11px; font-weight:600; margin-bottom:8px;
              text-transform:uppercase; letter-spacing:1px;'>
        📋 Feature Reference Guide
    </p>""", unsafe_allow_html=True)

    guide_df = pd.DataFrame({
        "Feature"      : ["MMSE","Functional Assessment",
                           "ADL","Memory Complaints",
                           "Behavioral Problems"],
        "Range"        : ["0–30","0–10","0–10","0 or 1","0 or 1"],
        "Normal Range" : ["24–30","7–10","7–10","No (0)","No (0)"],
        "Risk Range"   : ["< 24","< 5","< 5","Yes (1)","Yes (1)"],
        "Model Weight" : [f"{feat_imp.get(f,0):.3f}"
                          for f in features],
    })
    st.dataframe(guide_df, hide_index=True, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:20px 0 12px;
            border-top:1px solid #E8EBF0; margin-top:12px;'>
    <span style='color:#9CA3AF; font-size:12px;'>
        🧠 <b style='color:#6B7280;'>NeuroScan AI</b>
        &nbsp;·&nbsp; SVM — Top 5 Cognitive Features
        &nbsp;·&nbsp; SMOTE Balanced
        &nbsp;·&nbsp; For Educational Use Only
    </span>
</div>
""", unsafe_allow_html=True)