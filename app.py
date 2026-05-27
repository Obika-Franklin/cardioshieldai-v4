# app.py - CardioShield AI Streamlit App
# Professional icon set via Font Awesome 6.5.1

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
import sqlite3
import base64
import time
from PIL import Image
from io import BytesIO
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CardioShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS THEME + FONT AWESOME
# ============================================================================

def inject_custom_css():
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', unsafe_allow_html=True)
    
    st.markdown("""
    <style>
        :root {
            --navy: #0B1F3A;
            --navy-hover: #122b4d;
            --teal: #2EC4B6;
            --bg: #F7F9FC;
            --card: #FFFFFF;
            --border: #E2E8F0;
            --text-primary: #1F2937;
            --text-secondary: #6B7280;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }
        
        .stApp {
            background-color: #F7F9FC;
            margin-top: -60px;
        }
        
        .stApp > header {
            background: transparent;
        }
        
        div[data-testid="stToolbar"] {
            display: none;
        }
        
        div[data-testid="stDecoration"] {
            display: none;
        }
        
        div[data-testid="stStatusWidget"] {
            display: none;
        }
        
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        .custom-header {
            background: linear-gradient(135deg, #0B1F3A 0%, #122b4d 100%);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 2rem;
            border-radius: 0 0 16px 16px;
        }
        
        .custom-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .risk-badge-high {
            background: #FEE2E2;
            color: #EF4444;
            border: 1px solid #FECACA;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin: 8px 0;
        }
        
        .risk-badge-moderate {
            background: #FEF3C7;
            color: #F59E0B;
            border: 1px solid #FDE68A;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin: 8px 0;
        }
        
        .risk-badge-low {
            background: #D1FAE5;
            color: #10B981;
            border: 1px solid #A7F3D0;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
            margin: 8px 0;
        }
        
        .kpi-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
            transition: background 0.2s;
        }
        
        .kpi-card:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .result-card {
            background: white;
            border-radius: 20px;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .empty-state {
            border: 2px dashed #E2E8F0;
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            background: rgba(247, 249, 252, 0.5);
        }

        .alert-warning {
            background: rgba(255, 251, 235, 0.8);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 12px;
            padding: 1rem;
            color: #92400E;
        }
        
        .alert-success {
            background: #D1FAE5;
            border: 1px solid #A7F3D0;
            border-radius: 12px;
            padding: 1rem;
            color: #065F46;
        }
        
        .alert-error {
            background: #FEE2E2;
            border: 1px solid #FECACA;
            border-radius: 12px;
            padding: 1rem;
            color: #991B1B;
        }

        .stButton > button {
            background: linear-gradient(135deg, #0B1F3A 0%, #122b4d 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 700;
            transition: all 0.2s;
            box-shadow: 0 2px 8px rgba(11, 31, 58, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(11, 31, 58, 0.3);
        }
        
        .investor-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .investor-card-red {
            border-top: 4px solid #EF4444;
        }
        
        .investor-card-teal {
            border-top: 4px solid #2EC4B6;
        }
        
        .icon-circle {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .icon-circle-red {
            background: #FEE2E2;
        }
        
        .icon-circle-teal {
            background: #D1FAE5;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: white;
            border-radius: 16px;
            padding: 0.5rem;
            border: 1px solid #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 12px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            color: #6B7280;
        }
        
        .stTabs [aria-selected="true"] {
            background: #0B1F3A;
            color: white;
        }
        
        .sidebar-metric {
            background: #F7F9FC;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin-bottom: 16px;
        }
        
        .model-status {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 9999px;
            font-size: 0.65rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .model-status-loaded {
            background: #D1FAE5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }
        
        .model-status-missing {
            background: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }
        
        .ecg-preview-container {
            background: #0a0f1a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1a2a3a;
            margin: 12px 0;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# ICON HELPERS
# ============================================================================

def icon(name, size="", color="", cls=""):
    """Generate Font Awesome icon HTML"""
    size_style = f"font-size:{size};" if size else ""
    color_style = f"color:{color};" if color else ""
    style = f'style="{size_style}{color_style}"' if (size_style or color_style) else ""
    return f'<i class="fa-solid {name} {cls}" {style}></i>'

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PREPROCESSOR_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/preprocessor/preprocessor.pkl"
RF_MODEL_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/rf_model/rf_model.pkl"
VGG16_MODEL_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/v1.0.0/vgg16_ecg_model.keras"

DEMO_ECG_URLS = {
    "normal": "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/normal-ecg/Normal.97.-.Copy.jpg",
    "myocardial_infarction": "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/myocardial-infarction/MI.99.-.Copy.jpg",
}

SAMPLE_LABELS = {
    "normal": "Normal Sinus Rhythm",
    "myocardial_infarction": "Myocardial Infarction (STEMI)",
}

@st.cache_resource
def download_file(url, filename):
    """Download a file with progress tracking"""
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        with st.spinner(f"Downloading {filename}..."):
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to download {filename}: {e}")
                return None
    return filepath

@st.cache_resource
def load_models():
    """Load or download all required models"""
    preprocessor = None
    rf_model = None
    vgg16_model = None
    
    try:
        preprocessor_path = download_file(PREPROCESSOR_URL, "preprocessor.pkl")
        rf_model_path = download_file(RF_MODEL_URL, "rf_model.pkl")
        vgg16_model_path = download_file(VGG16_MODEL_URL, "vgg16_ecg_model.keras")
        
        if preprocessor_path and rf_model_path:
            preprocessor = joblib.load(preprocessor_path)
            rf_model = joblib.load(rf_model_path)
        
        if vgg16_model_path:
            try:
                from tensorflow.keras.models import load_model
                vgg16_model = load_model(vgg16_model_path)
            except Exception:
                pass
        
    except Exception as e:
        st.error(f"Model loading failed: {e}")
    
    return preprocessor, rf_model, vgg16_model

@st.cache_data
def fetch_demo_ecg(sample_type):
    """Fetch demo ECG image from GitHub releases and return as base64"""
    url = DEMO_ECG_URLS.get(sample_type)
    if not url:
        return None
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode()
    except Exception as e:
        st.error(f"Failed to fetch demo ECG: {e}")
        return None

# ============================================================================
# FEATURE IMPORTANCE AGGREGATION
# ============================================================================

@st.cache_resource
def get_aggregated_feature_importance(_rf_model, _preprocessor):
    """Aggregate feature importance from 22 post-OHE features back to 11 original clinical features"""
    if _rf_model is None or _preprocessor is None:
        return None
    
    raw_importances = _rf_model.feature_importances_
    
    # Build mapping from OHE column names to original features
    ohe_names = _preprocessor.get_feature_names_out()
    
    # Define which OHE columns map to which original features
    feature_groups = {
        'age': ['age'],
        'resting bp s': ['resting bp s'],
        'cholesterol': ['cholesterol'],
        'max heart rate': ['max heart rate'],
        'oldpeak': ['oldpeak'],
        'sex': ['sex_0', 'sex_1'],
        'chest pain type': ['chest pain type_1', 'chest pain type_2', 'chest pain type_3', 'chest pain type_4'],
        'fasting blood sugar': ['fasting blood sugar_0', 'fasting blood sugar_1'],
        'resting ecg': ['resting ecg_0', 'resting ecg_1', 'resting ecg_2'],
        'exercise angina': ['exercise angina_0', 'exercise angina_1'],
        'ST slope': ['ST slope_0', 'ST slope_1', 'ST slope_2', 'ST slope_3'],
    }
    
    # Aggregate importances
    aggregated = {}
    for orig_name, ohe_variants in feature_groups.items():
        total_imp = 0.0
        for ohe_variant in ohe_variants:
            if ohe_variant in ohe_names:
                idx = list(ohe_names).index(ohe_variant)
                total_imp += float(raw_importances[idx])
        aggregated[orig_name] = total_imp
    
    return aggregated if any(v > 0 for v in aggregated.values()) else None

# ============================================================================
# DATABASE
# ============================================================================

@st.cache_resource
def init_db():
    conn = sqlite3.connect('waitlist.db', check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS waitlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def add_to_waitlist(name, email):
    conn = init_db()
    try:
        conn.execute('INSERT INTO waitlist (name, email) VALUES (?, ?)', (name, email))
        conn.commit()
        return conn.execute('SELECT COUNT(*) FROM waitlist').fetchone()[0]
    except sqlite3.IntegrityError:
        return None

def get_waitlist_count():
    return init_db().execute('SELECT COUNT(*) FROM waitlist').fetchone()[0]

# ============================================================================
# INFERENCE
# ============================================================================

def predict_rf(patient_data, preprocessor, rf_model):
    """RF prediction — requires models to be loaded"""
    if preprocessor is None or rf_model is None:
        raise RuntimeError("Models not loaded. Please wait for models to download.")
    
    df = pd.DataFrame([{
        'age': patient_data.get('age', 45),
        'sex': patient_data.get('sex', 1),
        'chest pain type': patient_data.get('chestPainType', 2),
        'resting bp s': patient_data.get('restingBpS', 130),
        'cholesterol': patient_data.get('cholesterol', 220),
        'fasting blood sugar': patient_data.get('fastingBloodSugar', 0),
        'resting ecg': patient_data.get('restingEcg', 0),
        'max heart rate': patient_data.get('maxHeartRate', 150),
        'exercise angina': patient_data.get('exerciseAngina', 0),
        'oldpeak': patient_data.get('oldpeak', 0.0),
        'ST slope': patient_data.get('stSlope', 1),
    }])
    
    X_processed = preprocessor.transform(df)
    proba = rf_model.predict_proba(X_processed)[0]
    risk_score = proba[1] * 100
    
    if risk_score >= 70:
        risk_level = "high"
        recommendation = "Patient exhibits multiple cardiovascular risk factors. Immediate cardiology referral recommended. Consider stress testing and comprehensive lipid panel."
    elif risk_score >= 30:
        risk_level = "moderate"
        recommendation = "Moderate cardiovascular risk detected. Monitor patient closely and consider lifestyle interventions. Follow-up in 3-6 months with repeat assessment."
    else:
        risk_level = "low"
        recommendation = "Low cardiovascular risk profile. Continue routine preventive care. Maintain healthy lifestyle and schedule annual check-up."
    
    agg_importance = get_aggregated_feature_importance(rf_model, preprocessor)
    st.write("DEBUG - agg_importance:", agg_importance)
    features = []
    if agg_importance:
        features = [{"name": k, "importance": v} for k, v in agg_importance.items()]
        features.sort(key=lambda x: x["importance"], reverse=True)
    
    return {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "rfProbability": proba[1],
        "recommendation": recommendation,
        "modelAccuracy": 0.9202,
        "features": features
    }

def predict_ecg(image_data, vgg16_model):
    """VGG16 ECG classification — requires model to be loaded"""
    if vgg16_model is None:
        raise RuntimeError("VGG16 model not loaded. Please wait for model to download.")
    
    from tensorflow.keras.preprocessing import image as keras_image
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.resize((100, 100)).convert('RGB')
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = vgg16_model.predict(img_array, verbose=0)[0]
    classes = ['abnormal_heartbeat', 'history_mi', 'myocardial_infarction', 'normal']
    
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx])
    classification = classes[predicted_idx]
    
    findings_map = {
        'normal': "Normal sinus rhythm. Regular P-QRS-T waveform pattern. No ST segment abnormalities detected. Heart rate within normal range.",
        'myocardial_infarction': "ST segment elevation detected. Pathological Q waves present. Findings consistent with acute myocardial infarction. Urgent cardiology evaluation required.",
        'history_mi': "Residual Q waves detected. T-wave inversion noted. Findings suggest prior myocardial infarction. Continued monitoring and follow-up recommended.",
        'abnormal_heartbeat': "Irregular rhythm pattern detected. Varying QRS amplitudes. Abnormal heartbeat morphology. Further diagnostic workup advised."
    }
    
    risk_map = {'normal': 'low', 'myocardial_infarction': 'high', 'history_mi': 'moderate', 'abnormal_heartbeat': 'moderate'}
    
    return {
        "classification": classification.replace('_', ' ').title(),
        "confidence": confidence,
        "findings": findings_map.get(classification, ""),
        "riskLevel": risk_map.get(classification, "moderate"),
        "modelAccuracy": 0.7483,
        "modelName": "CNN VGG16",
        "probabilities": {classes[i]: float(predictions[i]) for i in range(len(classes))}
    }

def combine_results(rf_result, ecg_result=None):
    if ecg_result is None:
        return {
            "rfResult": rf_result,
            "ecgProvided": False,
            "finalRiskLevel": rf_result["riskLevel"],
            "finalRecommendation": rf_result["recommendation"],
            "confidenceScore": rf_result["rfProbability"]
        }
    
    risk_map = {"low": 1, "moderate": 2, "high": 3}
    reverse_map = {1: "low", 2: "moderate", 3: "high"}
    
    combined_score = risk_map[rf_result["riskLevel"]] * 0.6 + risk_map[ecg_result["riskLevel"]] * 0.4
    final_risk_level = reverse_map[round(combined_score)]
    confidence_score = (rf_result["rfProbability"] + ecg_result["confidence"]) / 2
    
    recommendations = {
        "high": "Combined RF + VGG16 analysis indicates HIGH cardiovascular risk. Urgent cardiology referral required. Both structured vitals and ECG morphology suggest significant pathology. Immediate clinical action recommended.",
        "moderate": "Combined analysis indicates MODERATE cardiovascular risk. Further diagnostic testing recommended. Monitor patient symptoms and schedule follow-up within 1-3 months.",
        "low": "Combined analysis indicates LOW cardiovascular risk. Routine preventive care recommended. Both models agree on low-risk classification. Continue annual check-ups."
    }
    
    return {
        "rfResult": rf_result,
        "ecgResult": ecg_result,
        "ecgProvided": True,
        "finalRiskLevel": final_risk_level,
        "finalRecommendation": recommendations[final_risk_level],
        "confidenceScore": confidence_score
    }

# ============================================================================
# CHART RENDERERS
# ============================================================================

def render_risk_gauge(score, risk_level):
    colors = {"low": ("#2EC4B6", "#E6F7F5"), "moderate": ("#F59E0B", "#FEF3C7"), "high": ("#EF4444", "#FEE2E2")}
    gauge_color, _ = colors.get(risk_level, colors["low"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk", 'font': {'size': 14, 'color': '#0B1F3A'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': '#0B1F3A', 'family': 'Segoe UI'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#9CA3AF"},
            'bar': {'color': gauge_color, 'thickness': 0.15},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 30], 'color': '#E6F7F5'},
                {'range': [30, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#FEE2E2'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20),
                      paper_bgcolor='rgba(0,0,0,0)', font={'color': '#0B1F3A', 'family': 'Segoe UI'})
    return fig

def render_feature_importance_chart(features):
    if not features: return None
    df = pd.DataFrame(features).sort_values('importance', ascending=True).tail(10)
    fig = px.bar(df, x='importance', y='name', orientation='h',
                 title='Feature Importance (Random Forest)',
                 color='importance', color_continuous_scale=['#E6F7F5', '#2EC4B6', '#0B1F3A'])
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': '#0B1F3A'}, xaxis_title="Importance", yaxis_title="")
    return fig

def render_ecg_probability_chart(probabilities):
    if not probabilities: return None
    items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace('_', ' ').title() for k, v in items]
    values = [v * 100 for k, v in items]
    
    fig = go.Figure()
    for i, (label, value) in enumerate(zip(labels, values)):
        fig.add_trace(go.Bar(y=[label], x=[value], orientation='h',
                            marker_color='#10B981' if i == 0 else '#CBD5E1',
                            text=f'{value:.1f}%', textposition='outside',
                            textfont={'color': '#0B1F3A', 'size': 12}))
    fig.update_layout(title='CNN Class Probabilities', height=200,
                      margin=dict(l=10, r=50, t=40, b=10), showlegend=False,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font={'color': '#0B1F3A'})
    return fig

def risk_badge_html(level):
    badges = {
        "high": f'<span class="risk-badge-high">{icon("fa-triangle-exclamation")} HIGH RISK</span>',
        "moderate": f'<span class="risk-badge-moderate">{icon("fa-circle-exclamation")} MODERATE RISK</span>',
        "low": f'<span class="risk-badge-low">{icon("fa-circle-check")} LOW RISK</span>'
    }
    return badges.get(level, badges["low"])

# ============================================================================
# PDF GENERATION
# ============================================================================

def generate_pdf_html(data):
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y, %I:%M %p")
    
    mode_labels = {"dual": "Dual Mode (RF + VGG16)", "ecg": "ECG-Only Mode (VGG16)", "data": "Data-Only Mode (RF + SMOTE)"}
    
    def risk_color(level):
        return {"high": "#D32F2F", "moderate": "#F57C00"}.get(level, "#2E7D32")
    
    def risk_label(level):
        return {"high": "HIGH RISK", "moderate": "MODERATE RISK"}.get(level, "LOW RISK")
    
    rf_section = ""
    if data.get("rfResult"):
        rf = data["rfResult"]
        rf_section = f'''
        <div class="section">
            <div class="section-title">Random Forest + SMOTE Analysis</div>
            <div class="model-badge">RF + SMOTE &bull; Accuracy: {rf.get("modelAccuracy", 0.92) * 100:.2f}%</div>
            <div class="risk-badge" style="background:{risk_color(rf["riskLevel"])}">
                {risk_label(rf["riskLevel"])} &mdash; Score: {rf["riskScore"]:.1f}/100
            </div>
            <div class="findings">{rf["recommendation"]}</div>
        </div>'''
    
    ecg_section = ""
    if data.get("ecgResult"):
        ecg = data["ecgResult"]
        ecg_section = f'''
        <div class="section">
            <div class="section-title">VGG16 ECG Classification</div>
            <div class="model-badge">CNN VGG16 &bull; Accuracy: {ecg.get("modelAccuracy", 0.75) * 100:.2f}%</div>
            <div class="risk-badge" style="background:{risk_color(ecg["riskLevel"])}">
                {ecg["classification"]} &mdash; Confidence: {ecg["confidence"] * 100:.1f}%
            </div>
            <div class="findings">{ecg["findings"]}</div>
        </div>'''
    
    combined_section = ""
    if data.get("mode") == "dual" and data.get("finalRiskLevel"):
        combined_section = f'''
        <div class="section combined">
            <div class="section-title">Combined Triage Assessment</div>
            <div class="risk-badge" style="background:{risk_color(data["finalRiskLevel"])}">
                {risk_label(data["finalRiskLevel"])} &mdash; Confidence: {(data.get("confidenceScore", 0) * 100):.1f}%
            </div>
            <div class="findings">{data.get("finalRecommendation", "")}</div>
        </div>'''
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>CardioShield AI &mdash; Clinical Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #212121; padding: 40px; }}
        .header {{ border-bottom: 3px solid #001F3F; padding-bottom: 16px; margin-bottom: 24px; }}
        .brand {{ font-size: 24px; font-weight: 700; color: #001F3F; }}
        .brand-sub {{ font-size: 12px; color: #26A69A; font-weight: 600; }}
        .mode-badge {{ display: inline-block; background: #001F3F; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 11px; margin-bottom: 20px; }}
        .section {{ background: #F5F5F5; border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #26A69A; }}
        .section.combined {{ border-left-color: #001F3F; }}
        .section-title {{ font-size: 14px; font-weight: 700; color: #001F3F; text-transform: uppercase; margin-bottom: 12px; }}
        .model-badge {{ display: inline-block; background: #26A69A; color: #fff; padding: 2px 10px; border-radius: 3px; font-size: 10px; margin-bottom: 10px; }}
        .risk-badge {{ display: inline-block; color: #fff; padding: 6px 16px; border-radius: 4px; font-size: 13px; font-weight: 700; margin-bottom: 12px; }}
        .findings {{ font-size: 13px; color: #424242; line-height: 1.7; }}
        .disclaimer {{ margin-top: 24px; padding: 14px; background: #FFF3E0; border-left: 4px solid #F57C00; border-radius: 4px; font-size: 11px; color: #5D4037; }}
        .footer {{ margin-top: 20px; padding-top: 12px; border-top: 1px solid #e0e0e0; font-size: 10px; color: #9E9E9E; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <div class="brand">CardioShield AI</div>
        <div class="brand-sub">CLINICAL DECISION SUPPORT REPORT</div>
        <div style="font-size: 11px; color: #757575; margin-top: 4px;">Generated: {date_str}</div>
    </div>
    <div class="mode-badge">{mode_labels.get(data.get("mode", "data"), "")}</div>
    {rf_section}
    {ecg_section}
    {combined_section}
    <div class="disclaimer"><strong>Clinical Disclaimer:</strong> CardioShield AI is a clinical decision support tool only. This report does NOT constitute a medical diagnosis.</div>
    <div class="footer">CardioShield AI &bull; RF+SMOTE 92.02% &bull; VGG16 74.83%</div>
</body>
</html>'''

def pdf_download_button(pdf_data, label="Download Clinical Report"):
    pdf_html = generate_pdf_html(pdf_data)
    b64 = base64.b64encode(pdf_html.encode()).decode()
    icon_html = icon("fa-file-pdf")
    href = f'<a href="data:text/html;base64,{b64}" download="CardioShield_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html"><button style="width:100%;margin-top:16px;padding:12px;background:#0B1F3A;color:white;border:none;border-radius:12px;font-weight:700;cursor:pointer;">{icon_html} {label}</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    inject_custom_css()
    
    preprocessor, rf_model, vgg16_model = load_models()
    
    rf_ready = preprocessor is not None and rf_model is not None
    vgg16_ready = vgg16_model is not None
    
    defaults = {
        'ecg_sample': None,
        'ecg_image': None,
        'rf_result': None,
        'ecg_result': None,
        'dual_result': None,
        'data_rf_result': None,
        'ecg_preview_image': None,  # Stores base64 of fetched demo ECG for preview
        'data_age_widget': 45,
        'data_sex_widget': 1,
        'data_chest_pain_widget': 2,
        'data_resting_bp_widget': 130,
        'data_cholesterol_widget': 220,
        'data_fbs_widget': False,
        'data_resting_ecg_widget': 0,
        'data_max_hr_widget': 150,
        'data_ex_angina_widget': False,
        'data_oldpeak_widget': 0.0,
        'data_st_slope_widget': 1,
        'dual_age_widget': 45,
        'dual_sex_widget': 1,
        'dual_chest_pain_widget': 2,
        'dual_resting_bp_widget': 130,
        'dual_cholesterol_widget': 220,
        'dual_fbs_widget': False,
        'dual_resting_ecg_widget': 0,
        'dual_max_hr_widget': 150,
        'dual_ex_angina_widget': False,
        'dual_oldpeak_widget': 0.0,
        'dual_st_slope_widget': 1,
        'data_preset': None,
        'dual_preset': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    header_shield = icon("fa-shield-halved", "24px", "#fff")
    header_bolt = icon("fa-bolt", "", "#2EC4B6")
    header_chart = icon("fa-chart-line", "", "#2EC4B6")
    header_clock = icon("fa-stopwatch", "", "#2EC4B6")
    header_db = icon("fa-database", "", "#2EC4B6")
    
    rf_status_class = "model-status-loaded" if rf_ready else "model-status-missing"
    rf_status_text = "RF LOADED" if rf_ready else "RF MISSING"
    vgg_status_class = "model-status-loaded" if vgg16_ready else "model-status-missing"
    vgg_status_text = "VGG16 LOADED" if vgg16_ready else "VGG16 MISSING"
    
    st.markdown(f"""
    <div class="custom-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #2EC4B6, #25a99d); 
                     border-radius: 12px; display: flex; align-items: center; justify-content: center; 
                     box-shadow: 0 4px 12px rgba(46, 196, 182, 0.3);">
                    {header_shield}
                </div>
                <div>
                    <h1 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">CardioShield AI</h1>
                    <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.8rem;">
                        Cardiovascular Risk Prediction
                        <span class="model-status {rf_status_class}" style="margin-left:8px;">{rf_status_text}</span>
                        <span class="model-status {vgg_status_class}" style="margin-left:4px;">{vgg_status_text}</span>
                    </p>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 12px; margin-top: 20px; overflow-x: auto;">
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="padding:8px;background:rgba(46,196,182,0.15);border-radius:8px;">{header_bolt}</div>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">RF + SMOTE</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">1,200 patient records</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="padding:8px;background:rgba(46,196,182,0.15);border-radius:8px;">{header_chart}</div>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">VGG16 ECG</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">2,500 ECG images</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="padding:8px;background:rgba(46,196,182,0.15);border-radius:8px;">{header_clock}</div>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">Rapid</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">Inference response time</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="padding:8px;background:rgba(46,196,182,0.15);border-radius:8px;">{header_db}</div>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">3,700+</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">Combined data points</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if not rf_ready:
        st.error("RF model not loaded. Predictions will fail. Check model download URLs.")
    if not vgg16_ready:
        st.warning("VGG16 model not loaded. ECG predictions (including demo samples) will fail.")
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dual Mode",
        "ECG-Only",
        "Data-Only",
        "Investor Brief"
    ])
    
    # ========================================================================
    # TAB 1: DUAL MODE
    # ========================================================================
    
    with tab1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
            <span style="padding:4px 12px;border-radius:8px;border:1.5px solid rgba(46,196,182,0.35);
                  background:rgba(46,196,182,0.06);color:#0B1F3A;font-size:11px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;">
                  {icon("fa-microscope", "", "#2EC4B6")} Dual Mode &middot; RF + CNN VGG16</span>
            <span style="color:#6B7280;font-size:0.85rem;">Patient vitals + ECG combined for maximum triage accuracy</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Patient Vitals")
            patient_name = st.text_input("Patient ID / Name", key="dual_patient_name")
            
            if st.session_state.get('dual_preset') == 'low':
                st.session_state.dual_age_widget = 40
                st.session_state.dual_sex_widget = 1
                st.session_state.dual_chest_pain_widget = 2
                st.session_state.dual_resting_bp_widget = 140
                st.session_state.dual_cholesterol_widget = 289
                st.session_state.dual_fbs_widget = False
                st.session_state.dual_resting_ecg_widget = 0
                st.session_state.dual_max_hr_widget = 172
                st.session_state.dual_ex_angina_widget = False
                st.session_state.dual_oldpeak_widget = 0.0
                st.session_state.dual_st_slope_widget = 1
                st.session_state.dual_preset = None
            elif st.session_state.get('dual_preset') == 'high':
                st.session_state.dual_age_widget = 49
                st.session_state.dual_sex_widget = 0
                st.session_state.dual_chest_pain_widget = 3
                st.session_state.dual_resting_bp_widget = 160
                st.session_state.dual_cholesterol_widget = 180
                st.session_state.dual_fbs_widget = False
                st.session_state.dual_resting_ecg_widget = 0
                st.session_state.dual_max_hr_widget = 156
                st.session_state.dual_ex_angina_widget = False
                st.session_state.dual_oldpeak_widget = 1.0
                st.session_state.dual_st_slope_widget = 2
                st.session_state.dual_preset = None
            
            fc1, fc2 = st.columns(2)
            with fc1:
                age = st.number_input("Age (years)", 18, 100, key="dual_age_widget")
                resting_bp = st.number_input("Resting BP (mmHg)", 60, 250, key="dual_resting_bp_widget")
                cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, key="dual_cholesterol_widget")
                max_hr = st.number_input("Max Heart Rate (bpm)", 50, 250, key="dual_max_hr_widget")
                oldpeak = st.number_input("ST Depression (Oldpeak)", -3.0, 7.0, step=0.1, key="dual_oldpeak_widget")
            with fc2:
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="dual_sex_widget")
                chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                    format_func=lambda x: {1:"1 — Typical Angina", 2:"2 — Atypical Angina", 3:"3 — Non-Anginal", 4:"4 — Asymptomatic"}[x], key="dual_chest_pain_widget")
                resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: {0:"0 — Normal", 1:"1 — ST-T Wave", 2:"2 — LV Hypertrophy"}[x], key="dual_resting_ecg_widget")
                st_slope = st.selectbox("ST Slope", [1, 2, 3],
                    format_func=lambda x: {1:"1 — Upsloping", 2:"2 — Flat", 3:"3 — Downsloping"}[x], key="dual_st_slope_widget")
                fbs = st.toggle("Fasting Blood Sugar >120", key="dual_fbs_widget")
                ex_angina = st.toggle("Exercise Induced Angina", key="dual_ex_angina_widget")
            
            qc1, qc2 = st.columns(2)
            with qc1:
                if st.button("Load Low Risk", width="stretch", key="dual_load_low"):
                    st.session_state.dual_preset = "low"
                    st.rerun()
            with qc2:
                if st.button("Load High Risk", width="stretch", key="dual_load_high"):
                    st.session_state.dual_preset = "high"
                    st.rerun()
            
            if st.button("Analyze Patient Data", type="primary", width="stretch", key="dual_analyze_btn", disabled=not rf_ready):
                patient_data = {
                    "patientName": patient_name, "age": age, "sex": sex,
                    "chestPainType": chest_pain, "restingBpS": resting_bp,
                    "cholesterol": cholesterol, "fastingBloodSugar": 1 if fbs else 0,
                    "restingEcg": resting_ecg, "maxHeartRate": max_hr,
                    "exerciseAngina": 1 if ex_angina else 0, "oldpeak": oldpeak, "stSlope": st_slope
                }
                with st.spinner("Running dual-model analysis..."):
                    time.sleep(0.5)
                    try:
                        st.session_state.rf_result = predict_rf(patient_data, preprocessor, rf_model)
                        if st.session_state.ecg_image and vgg16_ready:
                            st.session_state.ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                            st.session_state.dual_result = combine_results(st.session_state.rf_result, st.session_state.ecg_result)
                        elif st.session_state.ecg_sample and vgg16_ready:
                            demo_image = fetch_demo_ecg(st.session_state.ecg_sample)
                            if demo_image:
                                st.session_state.ecg_result = predict_ecg(demo_image, vgg16_model)
                                st.session_state.dual_result = combine_results(st.session_state.rf_result, st.session_state.ecg_result)
                        elif not vgg16_ready and (st.session_state.ecg_image or st.session_state.ecg_sample):
                            st.error("VGG16 model not loaded. Cannot classify ECG.")
                            st.session_state.dual_result = combine_results(st.session_state.rf_result, None)
                            st.session_state.ecg_result = None
                        else:
                            st.session_state.dual_result = combine_results(st.session_state.rf_result, None)
                            st.session_state.ecg_result = None
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
        
        with col2:
            st.markdown("#### ECG Image Scanner")
            
            sample_options = [None, "normal", "myocardial_infarction"]
            sample_labels = ["None", "Normal Sinus Rhythm", "Myocardial Infarction (STEMI)"]
            
            current_idx = sample_options.index(st.session_state.ecg_sample) if st.session_state.ecg_sample in sample_options else 0
            
            selected = st.selectbox("Select demo sample or upload below", range(len(sample_options)),
                                    format_func=lambda i: sample_labels[i], index=current_idx, key="dual_sample_select")
            
            if sample_options[selected] is not None:
                new_sample = sample_options[selected]
                # Fetch real image if sample changed
                if st.session_state.ecg_sample != new_sample:
                    st.session_state.ecg_sample = new_sample
                    st.session_state.ecg_image = None
                    st.session_state.ecg_preview_image = fetch_demo_ecg(new_sample)
                # Show real ECG image preview
                if st.session_state.ecg_preview_image:
                    try:
                        img_bytes = base64.b64decode(st.session_state.ecg_preview_image)
                        st.image(img_bytes, caption=f"Demo: {SAMPLE_LABELS.get(st.session_state.ecg_sample, '')}", width='stretch')
                    except Exception:
                        st.caption(f"Demo sample selected: {SAMPLE_LABELS.get(st.session_state.ecg_sample, '')}")
                if not vgg16_ready:
                    st.warning("VGG16 model not loaded. Demo sample cannot be classified.")
            else:
                st.session_state.ecg_sample = None
                st.session_state.ecg_preview_image = None
            
            uploaded_file = st.file_uploader("Upload ECG Image (PNG, JPG up to 20MB)", type=["png", "jpg", "jpeg"], key="dual_ecg_upload")
            if uploaded_file:
                st.session_state.ecg_sample = None
                st.session_state.ecg_preview_image = None
                st.session_state.ecg_image = base64.b64encode(uploaded_file.getvalue()).decode()
                st.image(uploaded_file, caption="Uploaded ECG", width='stretch')
                if not vgg16_ready:
                    st.warning("VGG16 model not loaded. Uploaded ECG cannot be classified.")
            
            if not st.session_state.ecg_sample and not st.session_state.ecg_image:
                st.info("No ECG selected — RF + SMOTE will still run on patient vitals. Select a sample or upload an ECG to enable dual-model triage.")
        
        # Results
        if st.session_state.dual_result:
            st.markdown("---")
            st.markdown("### Results")
            dual = st.session_state.dual_result
            
            if not dual.get("ecgProvided"):
                st.warning("ECG was not provided — results show Random Forest analysis only. Add an ECG input for dual-model combined triage.")
            
            has_ecg = dual.get("ecgProvided", False)
            if has_ecg:
                rc1, rc2 = st.columns([1, 1])
            else:
                rc1 = st.columns([1])[0]
                rc2 = None
            
            with rc1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Random Forest + SMOTE")
                st.caption("Structured patient vitals — 11 clinical features")
                
                rf = dual["rfResult"]
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                st.plotly_chart(render_risk_gauge(rf["riskScore"], rf["riskLevel"]), use_container_width=True, key="dual_rf_gauge")
                if rf.get("features"):
                    fig = render_feature_importance_chart(rf["features"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="dual_rf_features")
                st.markdown(f"""
                <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin-top:12px;">
                    <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">
                        {icon("fa-circle-info", "", "#2EC4B6")} Clinical Interpretation</p>
                    <p style="font-size:0.85rem;color:#4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if has_ecg and dual.get("ecgResult") and rc2 is not None:
                with rc2:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("#### CNN VGG16 ECG")
                    st.caption("Waveform morphology classification")
                    
                    ecg = dual["ecgResult"]
                    is_normal = ecg["riskLevel"] == "low"
                    conf_pct = int(ecg["confidence"] * 100)
                    
                    result_icon = icon("fa-circle-check", "2rem", "#10B981") if is_normal else icon("fa-triangle-exclamation", "2rem", "#EF4444")
                    
                    st.markdown(f"""
                    <div style="border-radius:16px;padding:16px;border:1px solid {'#D1FAE5' if is_normal else '#FEE2E2'};
                         background:{'#ECFDF5' if is_normal else '#FEF2F2'};margin:12px 0;">
                        <div style="display:flex;align-items:center;gap:12px;">
                            {result_icon}
                            <div>
                                <h4 style="color:#0B1F3A;margin:0;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                                <p style="color:#6B7280;font-size:0.8rem;margin:0;">{ecg["classification"]}</p>
                            </div>
                        </div>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin:12px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:0.7rem;font-weight:700;color:#6B7280;text-transform:uppercase;">Model Confidence</span>
                            <span style="font-size:1.5rem;font-weight:800;color:#0B1F3A;">{conf_pct}%</span>
                        </div>
                        <div style="height:12px;background:#E2E8F0;border-radius:999px;overflow:hidden;margin-top:8px;">
                            <div style="width:{conf_pct}%;height:100%;background:{'#10B981' if is_normal else '#EF4444'};border-radius:999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if ecg.get("probabilities"):
                        fig = render_ecg_probability_chart(ecg["probabilities"])
                        if fig: st.plotly_chart(fig, use_container_width=True, key="dual_ecg_probs")
                    
                    st.markdown(f"""
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin-top:12px;">
                        <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">
                            {icon("fa-magnifying-glass-chart", "", "#2EC4B6")} Key Findings</p>
                        <p style="font-size:0.85rem;color:#4B5563;">{ecg["findings"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Combined result
            frl = dual["finalRiskLevel"]
            colors = {"high": ("#EF4444", "#FEF2F2"), "moderate": ("#F59E0B", "#FFF7ED"), "low": ("#2EC4B6", "#ECFDF5")}
            icons_map = {
                "high": icon("fa-triangle-exclamation", "2rem", "#EF4444"),
                "moderate": icon("fa-circle-exclamation", "2rem", "#F59E0B"),
                "low": icon("fa-circle-check", "2rem", "#2EC4B6")
            }
            
            st.markdown(f"""
            <div style="border:2px solid {colors[frl][0]};border-radius:16px;padding:24px;background:{colors[frl][1]};margin-top:16px;">
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                    {icons_map[frl]}
                    <div>
                        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:#6B7280;letter-spacing:0.1em;">Consensus Triage</span>
                        <h3 style="margin:4px 0;color:#0B1F3A;text-transform:capitalize;font-size:1.5rem;">{frl} Risk</h3>
                        <span style="font-size:0.75rem;color:#6B7280;">{(dual["confidenceScore"] * 100):.1f}% Confidence</span>
                    </div>
                </div>
                <div style="background:white;padding:16px;border-radius:12px;">
                    <p style="color:#4B5563;font-size:0.9rem;">{dual["finalRecommendation"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            pdf_data = {
                "mode": "dual", "rfResult": dual["rfResult"],
                "ecgResult": dual.get("ecgResult"),
                "finalRiskLevel": dual["finalRiskLevel"],
                "finalRecommendation": dual["finalRecommendation"],
                "confidenceScore": dual["confidenceScore"]
            }
            pdf_download_button(pdf_data, "Download Clinical Report (PDF)")
    
    # ========================================================================
    # TAB 2: ECG-ONLY
    # ========================================================================
    
    with tab2:
        st.markdown("#### ECG Image Analysis")
        st.caption("CNN VGG16 waveform classification from ECG images — real model inference only")
        
        ec1, ec2 = st.columns([1, 1])
        
        with ec1:
            sample_options = [None, "normal", "myocardial_infarction"]
            sample_labels = ["None", "Normal Sinus Rhythm", "Myocardial Infarction (STEMI)"]
            
            current_idx = sample_options.index(st.session_state.ecg_sample) if st.session_state.ecg_sample in sample_options else 0
            
            selected = st.selectbox("Select demo sample", range(len(sample_options)),
                                    format_func=lambda i: sample_labels[i], index=current_idx, key="ecg_only_sample")
            
            if sample_options[selected] is not None:
                new_sample = sample_options[selected]
                if st.session_state.ecg_sample != new_sample:
                    st.session_state.ecg_sample = new_sample
                    st.session_state.ecg_image = None
                    st.session_state.ecg_preview_image = fetch_demo_ecg(new_sample)
                # Show real ECG image preview
                if st.session_state.ecg_preview_image:
                    try:
                        img_bytes = base64.b64decode(st.session_state.ecg_preview_image)
                        st.image(img_bytes, caption=f"Demo: {SAMPLE_LABELS.get(st.session_state.ecg_sample, '')}", width='stretch')
                    except Exception:
                        st.caption(f"Demo sample selected: {SAMPLE_LABELS.get(st.session_state.ecg_sample, '')}")
                if not vgg16_ready:
                    st.warning("VGG16 model not loaded. Demo sample cannot be classified.")
            else:
                st.session_state.ecg_sample = None
                st.session_state.ecg_preview_image = None
            
            st.markdown("---")
            st.caption("or upload an ECG image")
            uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"], key="ecg_only_upload")
            if uploaded_file:
                st.session_state.ecg_sample = None
                st.session_state.ecg_preview_image = None
                st.session_state.ecg_image = base64.b64encode(uploaded_file.getvalue()).decode()
                st.image(uploaded_file, caption="Uploaded ECG", width='stretch')
                if not vgg16_ready:
                    st.warning("VGG16 model not loaded. Uploaded ECG cannot be classified.")
            
            has_ecg = bool(st.session_state.ecg_sample or st.session_state.ecg_image)
            can_analyze = has_ecg and vgg16_ready
            
            if st.button("Analyze ECG", type="primary", width="stretch", disabled=not can_analyze, key="ecg_only_analyze"):
                if st.session_state.ecg_image and vgg16_ready:
                    with st.spinner("Processing ECG image with CNN VGG16..."):
                        time.sleep(0.5)
                        try:
                            st.session_state.ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                        except Exception as e:
                            st.error(f"ECG prediction failed: {e}")
                elif st.session_state.ecg_sample and vgg16_ready:
                    with st.spinner("Fetching demo ECG and running CNN VGG16..."):
                        time.sleep(0.5)
                        demo_image = fetch_demo_ecg(st.session_state.ecg_sample)
                        if demo_image:
                            try:
                                st.session_state.ecg_result = predict_ecg(demo_image, vgg16_model)
                            except Exception as e:
                                st.error(f"ECG prediction failed: {e}")
        
        with ec2:
            if st.session_state.ecg_result:
                ecg = st.session_state.ecg_result
                is_normal = ecg["riskLevel"] == "low"
                conf_pct = int(ecg["confidence"] * 100)
                
                result_icon = icon("fa-circle-check", "2rem", "#10B981") if is_normal else icon("fa-triangle-exclamation", "2rem", "#EF4444")
                
                st.markdown(f"""
                <div class="result-card">
                    <h4>Diagnostic Result</h4>
                    <p style="color:#6B7280;font-size:0.85rem;">CNN VGG16 Morphological Analysis</p>
                    <div style="border-radius:16px;padding:16px;border:1px solid {'#D1FAE5' if is_normal else '#FEE2E2'};
                         background:{'#ECFDF5' if is_normal else '#FEF2F2'};margin:12px 0;">
                        <div style="display:flex;align-items:center;gap:12px;">
                            {result_icon}
                            <div>
                                <h4 style="color:#0B1F3A;margin:0;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                                <p style="color:#6B7280;font-size:0.8rem;margin:0;">{ecg["classification"]}</p>
                            </div>
                        </div>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin:12px 0;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:0.7rem;font-weight:700;color:#6B7280;text-transform:uppercase;">Model Confidence</span>
                            <span style="font-size:1.5rem;font-weight:800;color:#0B1F3A;">{conf_pct}%</span>
                        </div>
                        <div style="height:12px;background:#E2E8F0;border-radius:999px;overflow:hidden;margin-top:8px;">
                            <div style="width:{conf_pct}%;height:100%;background:{'#10B981' if is_normal else '#EF4444'};border-radius:999px;"></div>
                        </div>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;">
                        <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">
                            {icon("fa-magnifying-glass-chart", "", "#2EC4B6")} Key Findings</p>
                        <p style="font-size:0.85rem;color:#4B5563;">{ecg["findings"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if ecg.get("probabilities"):
                    fig = render_ecg_probability_chart(ecg["probabilities"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="ecg_only_probs")
                
                pdf_data = {"mode": "ecg", "ecgResult": ecg}
                pdf_download_button(pdf_data, "Download ECG Report (PDF)")
            else:
                empty_icon = icon("fa-heart-pulse", "3rem", "#2EC4B6")
                st.markdown(f"""
                <div class="empty-state">
                    {empty_icon}
                    <h4 style="color:#0B1F3A;margin-top:16px;">Awaiting ECG Input</h4>
                    <p style="color:#6B7280;">Select a demo sample or upload an ECG image on the left to begin.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: DATA-ONLY
    # ========================================================================
    
    with tab3:
        st.markdown("#### Data-Only Analysis")
        st.caption("Random Forest + SMOTE evaluation. No ECG required — works anywhere, instantly.")
        
        dc1, dc2 = st.columns([1.2, 1])
        
        with dc1:
            patient_name = st.text_input("Patient ID / Name", key="data_patient_name")
            
            if st.session_state.get('data_preset') == 'low':
                st.session_state.data_age_widget = 40
                st.session_state.data_sex_widget = 1
                st.session_state.data_chest_pain_widget = 2
                st.session_state.data_resting_bp_widget = 140
                st.session_state.data_cholesterol_widget = 289
                st.session_state.data_fbs_widget = False
                st.session_state.data_resting_ecg_widget = 0
                st.session_state.data_max_hr_widget = 172
                st.session_state.data_ex_angina_widget = False
                st.session_state.data_oldpeak_widget = 0.0
                st.session_state.data_st_slope_widget = 1
                st.session_state.data_preset = None
            elif st.session_state.get('data_preset') == 'high':
                st.session_state.data_age_widget = 49
                st.session_state.data_sex_widget = 0
                st.session_state.data_chest_pain_widget = 3
                st.session_state.data_resting_bp_widget = 160
                st.session_state.data_cholesterol_widget = 180
                st.session_state.data_fbs_widget = False
                st.session_state.data_resting_ecg_widget = 0
                st.session_state.data_max_hr_widget = 156
                st.session_state.data_ex_angina_widget = False
                st.session_state.data_oldpeak_widget = 1.0
                st.session_state.data_st_slope_widget = 2
                st.session_state.data_preset = None
            
            fc1, fc2 = st.columns(2)
            with fc1:
                age = st.number_input("Age (years)", 18, 100, key="data_age_widget")
                resting_bp = st.number_input("Resting BP (mmHg)", 60, 250, key="data_resting_bp_widget")
                cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, key="data_cholesterol_widget")
                max_hr = st.number_input("Max Heart Rate (bpm)", 50, 250, key="data_max_hr_widget")
                oldpeak = st.number_input("ST Depression (Oldpeak)", -3.0, 7.0, step=0.1, key="data_oldpeak_widget")
            with fc2:
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="data_sex_widget")
                chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                    format_func=lambda x: {1:"1 — Typical Angina", 2:"2 — Atypical Angina", 3:"3 — Non-Anginal", 4:"4 — Asymptomatic"}[x], key="data_chest_pain_widget")
                resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: {0:"0 — Normal", 1:"1 — ST-T Wave", 2:"2 — LV Hypertrophy"}[x], key="data_resting_ecg_widget")
                st_slope = st.selectbox("ST Slope", [1, 2, 3],
                    format_func=lambda x: {1:"1 — Upsloping", 2:"2 — Flat", 3:"3 — Downsloping"}[x], key="data_st_slope_widget")
                fbs = st.toggle("Fasting Blood Sugar >120", key="data_fbs_widget")
                ex_angina = st.toggle("Exercise Induced Angina", key="data_ex_angina_widget")
            
            qc1, qc2 = st.columns(2)
            with qc1:
                if st.button("Load Low Risk Sample", width="stretch", key="data_load_low"):
                    st.session_state.data_preset = "low"
                    st.rerun()
            with qc2:
                if st.button("Load High Risk Sample", width="stretch", key="data_load_high"):
                    st.session_state.data_preset = "high"
                    st.rerun()
            
            if st.button("Analyze Patient Data", type="primary", width="stretch", key="data_analyze_btn", disabled=not rf_ready):
                patient_data = {
                    "patientName": patient_name, "age": age, "sex": sex,
                    "chestPainType": chest_pain, "restingBpS": resting_bp,
                    "cholesterol": cholesterol, "fastingBloodSugar": 1 if fbs else 0,
                    "restingEcg": resting_ecg, "maxHeartRate": max_hr,
                    "exerciseAngina": 1 if ex_angina else 0, "oldpeak": oldpeak, "stSlope": st_slope
                }
                with st.spinner("Analyzing patient data with Random Forest + SMOTE..."):
                    time.sleep(0.5)
                    try:
                        st.session_state.data_rf_result = predict_rf(patient_data, preprocessor, rf_model)
                    except Exception as e:
                        st.error(f"RF prediction failed: {e}")
        
        with dc2:
            if st.session_state.data_rf_result:
                rf = st.session_state.data_rf_result
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Clinical Assessment")
                st.caption("Random Forest Feature Evaluation")
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                st.plotly_chart(render_risk_gauge(rf["riskScore"], rf["riskLevel"]), use_container_width=True, key="data_rf_gauge")
                if rf.get("features"):
                    fig = render_feature_importance_chart(rf["features"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="data_rf_features")
                st.markdown(f"""
                <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;">
                    <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">
                        {icon("fa-circle-info", "", "#2EC4B6")} Clinical Interpretation</p>
                    <p style="font-size:0.85rem;color:#4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                pdf_data = {"mode": "data", "rfResult": rf}
                pdf_download_button(pdf_data, "Download Clinical Report (PDF)")
            else:
                empty_icon = icon("fa-chart-bar", "3rem", "#2EC4B6")
                st.markdown(f"""
                <div class="empty-state">
                    {empty_icon}
                    <h4 style="color:#0B1F3A;margin-top:16px;">Awaiting Patient Data</h4>
                    <p style="color:#6B7280;">Fill out the patient vitals form and click analyze to generate a cardiovascular risk assessment.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: INVESTOR BRIEF
    # ========================================================================
    
    with tab4:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:40px;">
            <span style="display:inline-block;padding:4px 16px;border-radius:999px;background:rgba(46,196,182,0.1);
                  border:1px solid rgba(46,196,182,0.3);color:#2EC4B6;font-size:0.7rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;">
                  {icon("fa-chart-pie", "", "#2EC4B6")} Investment Brief &middot; 2026</span>
            <h1 style="color:#0B1F3A;font-size:2.5rem;font-weight:700;">Redefining Cardiovascular Care with AI</h1>
            <p style="color:#6B7280;font-size:1.1rem;max-width:700px;margin:0 auto;">
                A dual-mode cardiovascular screening platform deploying Random Forest with SMOTE and CNN VGG16 
                for instant triage in resource-constrained clinical environments.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(4)
        stats = [
            ("91%", "RF+SMOTE Accuracy", "1,200 patient records"),
            ("2,500+", "ECG Images", "CNN VGG16 Trained"),
            ("< 2s", "Inference Time", "Per prediction"),
            ("$4B+", "Addressable Market", "Global CDS software")
        ]
        for col, (v, l, s) in zip(cols, stats):
            with col:
                st.markdown(f"""
                <div style="text-align:center;padding:20px;background:#F7F9FC;border-radius:16px;border:1px solid #E2E8F0;">
                    <span style="font-size:2rem;font-weight:800;color:#0B1F3A;">{v}</span>
                    <p style="font-weight:600;color:#1F2937;margin:4px 0;">{l}</p>
                    <p style="font-size:0.75rem;color:#9CA3AF;">{s}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        ac1, ac2 = st.columns(2)
        
        with ac1:
            st.markdown(f"""
            <div class="investor-card investor-card-red">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                    <div class="icon-circle icon-circle-red">
                        {icon("fa-triangle-exclamation", "", "#EF4444")}
                    </div>
                    <h3 style="color:#1F2937;margin:0;">The Clinical Bottleneck</h3>
                </div>
                <p style="color:#4B5563;">Primary care and rural clinics cannot rely on ECG for every screening. A standard ECG requires equipment, trained technicians, and cardiologists to interpret.</p>
                <p style="color:#4B5563;margin-top:12px;">This delays triage, limits scalability, and forces high-risk patients to wait for specialist availability — leading to worse outcomes and higher systemic costs.</p>
                <div style="background:#FEF2F2;padding:12px;border-radius:8px;margin-top:16px;border:1px solid #FEE2E2;">
                    <p style="color:#991B1B;font-size:0.8rem;font-weight:500;">
                        {icon("fa-triangle-exclamation", "", "#EF4444")} Cardiovascular disease accounts for 32% of all global deaths — early triage directly saves lives.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with ac2:
            st.markdown(f"""
            <div class="investor-card investor-card-teal">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
                    <div class="icon-circle icon-circle-teal">
                        {icon("fa-diagram-project", "", "#2EC4B6")}
                    </div>
                    <h3 style="color:#1F2937;margin:0;">Dual-Mode AI Architecture</h3>
                </div>
                <p style="color:#4B5563;font-weight:600;">Works dynamically with or without ECG availability.</p>
                <div style="margin-top:16px;">
                    <div style="display:flex;gap:12px;margin-bottom:16px;">
                        <div style="width:24px;height:24px;background:#0B1F3A;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div>
                            <p style="font-weight:700;color:#1F2937;font-size:0.8rem;text-transform:uppercase;">Phase 1 — Structured Vitals</p>
                            <p style="color:#6B7280;font-size:0.8rem;">Random Forest + SMOTE on 11 clinical features. No ECG required. 91% accuracy on 1,200 records. Deployable in any clinic globally.</p>
                        </div>
                    </div>
                    <div style="display:flex;gap:12px;">
                        <div style="width:24px;height:24px;background:#2EC4B6;color:white;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div>
                            <p style="font-weight:700;color:#1F2937;font-size:0.8rem;text-transform:uppercase;">Phase 2 — ECG Validation</p>
                            <p style="color:#6B7280;font-size:0.8rem;">CNN VGG16 on 2,500 ECG images across 4 classes. Deep learning morphological analysis for definitive classification when equipment is present.</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:32px;">
            <p style="color:#2EC4B6;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;">Why CardioShield</p>
            <h2 style="color:#0B1F3A;">Platform Value Proposition</h2>
        </div>
        """, unsafe_allow_html=True)
        
        vp_cols = st.columns(4)
        vps = [
            ("fa-circle-check", "Works Without ECG", "Democratizes screening to any clinic capable of taking basic vitals and history."),
            ("fa-chart-simple", "ECG Validation Layer", "Seamlessly integrates deep learning morphological analysis when equipment is available."),
            ("fa-bolt", "Low-Cost Architecture", "Optimized inference pipelines minimize compute costs per prediction — scalable from day one."),
            ("fa-arrow-trend-up", "Fast Pilotability", "No hardware required to launch. Software-only integration accelerates B2B sales cycles.")
        ]
        for col, (icn, title, desc) in zip(vp_cols, vps):
            with col:
                st.markdown(f"""
                <div style="text-align:center;padding:24px;">
                    <div style="width:56px;height:56px;background:#F7F9FC;border-radius:16px;display:flex;align-items:center;justify-content:center;margin:0 auto 16px;">
                        {icon(icn, "1.5rem", "#2EC4B6")}
                    </div>
                    <p style="font-weight:700;color:#0B1F3A;margin-bottom:8px;">{title}</p>
                    <p style="font-size:0.8rem;color:#6B7280;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg, #0B1F3A 0%, #122b4d 100%);border-radius:24px;padding:48px;text-align:center;color:white;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-50px;right:-50px;width:200px;height:200px;background:rgba(46,196,182,0.2);border-radius:50%;filter:blur(40px);"></div>
            <div style="position:relative;z-index:1;">
                <h2 style="color:white;font-size:2rem;font-weight:700;margin-bottom:12px;">Deploy CardioShield AI at your facility</h2>
                <p style="color:rgba(255,255,255,0.7);font-size:1.1rem;margin-bottom:32px;">
                    Join leading clinics getting early access to our combined RF + CNN screening architecture.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("investor_waitlist"):
            wc1, wc2 = st.columns([3, 1])
            with wc1:
                wl_name = st.text_input("Full Name", placeholder="Dr. Jane Smith", key="investor_name")
                wl_email = st.text_input("Work Email", placeholder="you@hospital.org", key="investor_email")
            with wc2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Get Early Access", type="primary", width="stretch"):
                    if wl_name and wl_email:
                        pos = add_to_waitlist(wl_name, wl_email)
                        if pos: st.success(f"You're #{pos} on the waitlist! We'll be in touch soon.")
                        else: st.warning("This email is already registered on the waitlist.")
                    else:
                        st.error("Please fill in all fields to join the waitlist.")
    
    # ========================================================================
    # DISCLAIMER
    # ========================================================================
    
    with st.expander("Responsible AI & Clinical Use Disclaimer"):
        st.markdown("""
        CardioShield AI is a clinical decision support tool designed to assist healthcare professionals in evaluating cardiovascular risk. 
        It is **not a substitute** for professional medical diagnosis, advice, or treatment.
        
        The tabular model uses Random Forest with SMOTE trained on the Heart Statlog Cleveland Hungary Final dataset (1,200 patient records, 11 features). 
        The ECG classifier uses a VGG16 architecture trained on 2,500 ECG images. Both models provide statistical probabilities — clinicians must apply 
        professional judgment and full clinical context.
        """)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center;padding:20px;">
        <p style="color:#9CA3AF;font-size:0.75rem;">
            CardioShield AI — Clinical decision support only &nbsp;&middot;&nbsp; RF+SMOTE: 1,200 records &nbsp;&middot;&nbsp; VGG16: 2,500 ECG images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR WAITLIST
    # ========================================================================
    
    with st.sidebar:
        st.markdown("### Join Waitlist")
        waitlist_count = get_waitlist_count()
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <span style="font-size:2rem;font-weight:800;color:#0B1F3A;">{waitlist_count}</span>
            <p style="color:#6B7280;font-size:0.8rem;margin:0;">Total Signups</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("sidebar_waitlist"):
            sn = st.text_input("Full Name", key="sidebar_wl_name")
            se = st.text_input("Work Email", key="sidebar_wl_email")
            if st.form_submit_button("Join Waitlist", type="primary", width="stretch"):
                if sn and se:
                    p = add_to_waitlist(sn, se)
                    if p: st.success(f"You're #{p} on the list!")
                    else: st.warning("Already registered.")

if __name__ == "__main__":
    main()
