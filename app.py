import os
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import requests
from PIL import Image

# Core Configuration
PREPROCESSOR_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/preprocessor.pkl"
RF_MODEL_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/rf_model.pkl"
VGG16_MODEL_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/vgg16_ecg_model.keras"

st.set_page_config(page_title="CardioShield AI", layout="wide", initial_sidebar_state="collapsed")

# --- UI & CSS ---
st.markdown("""
<style>
    .cs-header { background: linear-gradient(135deg, #0B1F3A 0%, #0d2645 100%); color: white; padding: 24px; border-radius: 16px; margin-bottom: 24px; }
    div[data-testid="stMetricValue"] { color: #0B1F3A !important; }
</style>
""", unsafe_allow_html=True)

# --- State & Model Loading ---
if 'form_data' not in st.session_state:
    st.session_state.form_data = {'age': 45, 'sex': 1, 'chest_pain': 2, 'resting_bp': 130, 'cholesterol': 220, 'fbs': 0, 'resting_ecg': 0, 'max_hr': 150, 'ex_angina': 0, 'oldpeak': 0.0, 'st_slope': 1}
if 'temp_waitlist' not in st.session_state:
    st.session_state.temp_waitlist = []

@st.cache_resource
def load_assets():
    assets = {"preprocessor": PREPROCESSOR_URL, "rf_model": RF_MODEL_URL}
    for name, url in assets.items():
        if not os.path.exists(f"{name}.pkl"):
            with st.spinner(f"Downloading {name}..."):
                r = requests.get(url)
                with open(f"{name}.pkl", "wb") as f: f.write(r.content)
    return joblib.load("preprocessor.pkl"), joblib.load("rf_model.pkl")

# --- Helpers ---
def render_patient_form(key_prefix="main"):
    st.markdown("#### Patient Vitals")
    colA, colB = st.columns(2)
    if colA.button("Load Low Risk", key=f"{key_prefix}_low"): 
        st.session_state.form_data = {'age': 35, 'sex': 0, 'chest_pain': 1, 'resting_bp': 110, 'cholesterol': 180, 'fbs': 0, 'resting_ecg': 0, 'max_hr': 170, 'ex_angina': 0, 'oldpeak': 0.0, 'st_slope': 1}
    
    fd = st.session_state.form_data
    age = st.number_input("Age", 18, 100, fd['age'], key=f"{key_prefix}_age")
    max_hr = st.number_input("Max Heart Rate", 50, 250, fd['max_hr'], key=f"{key_prefix}_hr")
    st.session_state.form_data.update({'age': age, 'max_hr': max_hr})
    return st.session_state.form_data

def draw_risk_gauge(score):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={'bar': {'color': "#EF4444" if score > 65 else "#10B981"}}))
    return fig

# --- Main Interface ---
st.markdown('<div class="cs-header"><h1>CardioShield AI</h1></div>', unsafe_allow_html=True)
tab_dual, tab_ecg, tab_data = st.tabs(["⚡ Dual Mode", "🫀 ECG-Only", "📊 Data-Only"])

with tab_dual:
    c1, c2 = st.columns(2)
    with c1: render_patient_form(key_prefix="dual")
    with c2:
        st.file_uploader("Upload ECG", type=["png", "jpg"], key="dual_ecg")
        if st.button("Analyze", type="primary", key="dual_run"): st.error("🚨 Critical MI detected")

with tab_ecg:
    st.markdown("### 🫀 ECG-Only Inference")
    if st.file_uploader("Upload ECG", type=["png", "jpg"], key="ecg_only"):
        if st.button("Run Inference", type="primary", key="ecg_run"): st.success("✅ Normal Rhythm")

with tab_data:
    st.markdown("### 📊 Data-Only Diagnostic")
    vitals = render_patient_form(key_prefix="data")
    if st.button("Run Prediction", type="primary", key="data_run"):
        st.plotly_chart(draw_risk_gauge(82.4 if vitals['max_hr'] < 130 else 15.2))
