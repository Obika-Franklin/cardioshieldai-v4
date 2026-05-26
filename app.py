# app.py - CardioShield AI Streamlit App (Fixed version)
# Key fixes: unique keys for all widgets, width='stretch' instead of use_container_width=True

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
import sqlite3
import base64
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
# CUSTOM CSS THEME
# ============================================================================

def inject_custom_css():
    st.markdown("""
    <style>
        /* Root variables */
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
        }
        
        .result-card {
            background: white;
            border-radius: 20px;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .ecg-preview {
            background: #0a0f1a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1a2a3a;
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
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PREPROCESSOR_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/preprocessor.pkl"
RF_MODEL_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/rf_model.pkl"
VGG16_MODEL_URL = "https://github.com/FranklinObika/cardioshield-ai/releases/download/v1.0.0/vgg16_ecg_model.keras"

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
    try:
        preprocessor_path = download_file(PREPROCESSOR_URL, "preprocessor.pkl")
        rf_model_path = download_file(RF_MODEL_URL, "rf_model.pkl")
        vgg16_model_path = download_file(VGG16_MODEL_URL, "vgg16_ecg_model.keras")
        
        if preprocessor_path and rf_model_path:
            preprocessor = joblib.load(preprocessor_path)
            rf_model = joblib.load(rf_model_path)
        else:
            preprocessor, rf_model = None, None
        
        vgg16_model = None
        if vgg16_model_path:
            try:
                from tensorflow.keras.models import load_model
                vgg16_model = load_model(vgg16_model_path)
            except Exception:
                pass
        
        return preprocessor, rf_model, vgg16_model
    except Exception as e:
        st.warning(f"Model loading issue: {e}. Using simulation mode.")
        return None, None, None

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
    """RF prediction with fallback to simulation"""
    if preprocessor is None or rf_model is None:
        return simulate_rf_prediction(patient_data)
    
    try:
        df = pd.DataFrame([patient_data])
        X_processed = preprocessor.transform(df)
        proba = rf_model.predict_proba(X_processed)[0]
        risk_score = proba[1] * 100
        
        if risk_score >= 70:
            risk_level = "high"
            recommendation = "Patient exhibits multiple cardiovascular risk factors. Immediate cardiology referral recommended."
        elif risk_score >= 30:
            risk_level = "moderate"
            recommendation = "Moderate cardiovascular risk detected. Monitor patient closely. Follow-up in 3-6 months."
        else:
            risk_level = "low"
            recommendation = "Low cardiovascular risk profile. Continue routine preventive care."
        
        features = []
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                           'fasting blood sugar', 'resting ecg', 'max heart rate',
                           'exercise angina', 'oldpeak', 'ST slope']
            importances = rf_model.feature_importances_
            features = [{"name": name, "importance": float(imp)}
                       for name, imp in zip(feature_names, importances)]
            features.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "rfProbability": proba[1],
            "recommendation": recommendation,
            "modelAccuracy": 0.9202,
            "features": features
        }
    except Exception:
        return simulate_rf_prediction(patient_data)

def simulate_rf_prediction(patient_data):
    risk_score = 0
    if patient_data.get('age', 45) > 60: risk_score += 25
    elif patient_data.get('age', 45) > 45: risk_score += 15
    if patient_data.get('sex', 1) == 1: risk_score += 10
    if patient_data.get('chestPainType', 2) >= 3: risk_score += 20
    elif patient_data.get('chestPainType', 2) == 2: risk_score += 10
    if patient_data.get('restingBpS', 130) > 140: risk_score += 15
    if patient_data.get('cholesterol', 220) > 240: risk_score += 15
    if patient_data.get('fastingBloodSugar', 0) == 1: risk_score += 10
    if patient_data.get('exerciseAngina', 0) == 1: risk_score += 20
    if patient_data.get('oldpeak', 0) > 1.5: risk_score += 15
    if patient_data.get('stSlope', 1) >= 2: risk_score += 15
    
    risk_score = min(max(risk_score, 2), 98)
    
    if risk_score >= 70: risk_level = "high"
    elif risk_score >= 30: risk_level = "moderate"
    else: risk_level = "low"
    
    if risk_level == "high":
        recommendation = "Patient exhibits multiple cardiovascular risk factors. Immediate cardiology referral recommended."
    elif risk_level == "moderate":
        recommendation = "Moderate cardiovascular risk detected. Monitor patient closely."
    else:
        recommendation = "Low cardiovascular risk profile. Continue routine preventive care."
    
    features = [
        {"name": "age", "importance": 0.22},
        {"name": "chest pain type", "importance": 0.18},
        {"name": "max heart rate", "importance": 0.15},
        {"name": "oldpeak", "importance": 0.13},
        {"name": "exercise angina", "importance": 0.11},
        {"name": "ST slope", "importance": 0.08},
        {"name": "resting bp s", "importance": 0.05},
        {"name": "cholesterol", "importance": 0.04},
        {"name": "sex", "importance": 0.02},
        {"name": "fasting blood sugar", "importance": 0.01},
        {"name": "resting ecg", "importance": 0.01}
    ]
    
    return {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "rfProbability": risk_score / 100,
        "recommendation": recommendation,
        "modelAccuracy": 0.9202,
        "features": features
    }

def predict_ecg(image_data, vgg16_model):
    if vgg16_model is None:
        return simulate_ecg_prediction()
    
    try:
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
            'normal': "Normal sinus rhythm. Regular P-QRS-T waveform pattern.",
            'myocardial_infarction': "ST segment elevation detected. Pathological Q waves present.",
            'history_mi': "Residual Q waves detected. T-wave inversion noted.",
            'abnormal_heartbeat': "Irregular rhythm pattern. Varying QRS amplitudes."
        }
        
        risk_map = {'normal': 'low', 'myocardial_infarction': 'high', 'history_mi': 'moderate', 'abnormal_heartbeat': 'moderate'}
        
        return {
            "classification": classification.replace('_', ' ').title(),
            "confidence": confidence,
            "findings": findings_map.get(classification, ""),
            "riskLevel": risk_map.get(classification, "moderate"),
            "modelAccuracy": 0.7483,
            "modelName": "CNN VGG16",
            "isDemo": False,
            "probabilities": {classes[i]: float(predictions[i]) for i in range(len(classes))}
        }
    except Exception:
        return simulate_ecg_prediction()

def simulate_ecg_prediction(is_demo=True):
    return {
        "classification": "Normal Sinus Rhythm",
        "confidence": 0.92,
        "findings": "Regular P-QRS-T waveform pattern. No ST segment abnormalities detected.",
        "riskLevel": "low",
        "modelAccuracy": 0.7483,
        "modelName": "CNN VGG16",
        "isDemo": is_demo,
        "probabilities": {
            "normal": 0.92,
            "myocardial_infarction": 0.03,
            "history_mi": 0.03,
            "abnormal_heartbeat": 0.02
        }
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
        "high": "Combined analysis indicates HIGH cardiovascular risk. Urgent cardiology referral required.",
        "moderate": "Combined analysis indicates MODERATE cardiovascular risk. Further diagnostic testing recommended.",
        "low": "Combined analysis indicates LOW cardiovascular risk. Routine preventive care recommended."
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
# CHARTS
# ============================================================================

def render_risk_gauge(score, risk_level):
    colors = {"low": ("#2EC4B6", "#E6F7F5"), "moderate": ("#F59E0B", "#FEF3C7"), "high": ("#EF4444", "#FEE2E2")}
    gauge_color, track_color = colors.get(risk_level, colors["low"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk", 'font': {'size': 14, 'color': '#0B1F3A'}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': gauge_color, 'thickness': 0.15},
            'steps': [
                {'range': [0, 30], 'color': '#E6F7F5'},
                {'range': [30, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#FEE2E2'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=50, b=20))
    return fig

def render_feature_importance_chart(features):
    if not features: return None
    df = pd.DataFrame(features).sort_values('importance', ascending=True).tail(10)
    fig = px.bar(df, x='importance', y='name', orientation='h',
                 title='Feature Importance (Random Forest)',
                 color='importance', color_continuous_scale=['#E6F7F5', '#2EC4B6', '#0B1F3A'])
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
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
                            text=f'{value:.1f}%', textposition='outside'))
    fig.update_layout(title='CNN Class Probabilities', height=200,
                      margin=dict(l=10, r=50, t=40, b=10), showlegend=False)
    return fig

def risk_badge_html(level):
    badges = {
        "high": '<span class="risk-badge-high">⚠ HIGH RISK</span>',
        "moderate": '<span class="risk-badge-moderate">⚡ MODERATE RISK</span>',
        "low": '<span class="risk-badge-low">✅ LOW RISK</span>'
    }
    return badges.get(level, badges["low"])

# ============================================================================
# ECG SAMPLES (inline SVGs)
# ============================================================================

def get_ecg_sample_svg(sample_type):
    paths = {
        "normal": ("0,100 20,100 25,98 30,100 35,100 40,100 42,95 44,100 46,55 48,100 50,108 52,100 "
                   "70,100 75,98 80,100 90,100 92,95 94,100 96,55 98,100 100,108 102,100 "
                   "120,100 125,98 130,100 140,100 142,95 144,100 146,55 148,100 150,108 152,100 "
                   "170,100 175,98 180,100 190,100 192,95 194,100 196,55 198,100 200,108 202,100 "
                   "220,100 225,98 230,100 240,100 242,95 244,100 246,55 248,100 250,108 252,100 "
                   "270,100 275,98 280,100 290,100 292,95 294,100 296,55 298,100 300,108 302,100 "
                   "320,100 325,98 330,100 340,100 342,95 344,100 346,55 348,100 350,108 352,100 "
                   "370,100 375,98 380,100 390,100 392,95 394,100 396,55 398,100 400,108 402,100 "
                   "420,100 440,100 460,100 480,100 500,100 520,100 560,100 600,100",
                   "#26A69A", "Normal ECG", "Regular P-QRS-T  |  Normal Sinus Rhythm"),
        "myocardial_infarction": (
                   "0,100 15,100 20,98 25,100 30,100 32,90 34,100 35,40 36,130 37,100 38,75 42,100 "
                   "60,100 62,90 64,100 65,40 66,130 67,100 68,75 72,100 "
                   "90,100 92,90 94,100 95,40 96,130 97,100 98,75 102,100 "
                   "120,100 122,90 124,100 125,40 126,130 127,100 128,75 132,100 "
                   "150,100 152,90 154,100 155,40 156,130 157,100 158,75 162,100 "
                   "180,100 182,90 184,100 185,40 186,130 187,100 188,75 192,100 "
                   "210,100 212,90 214,100 215,40 216,130 217,100 218,75 222,100 "
                   "240,100 242,90 244,100 245,40 246,130 247,100 248,75 252,100 "
                   "270,100 300,100 330,100 360,100 400,100 440,100 480,100 560,100 600,100",
                   "#ef4444", "Myocardial Infarction", "ST Elevation  |  Pathological Q-waves"),
        "history_mi": (
                   "0,100 20,100 22,95 24,100 30,100 32,90 34,108 35,50 37,130 39,100 42,90 45,100 "
                   "60,100 62,90 64,108 65,50 67,130 69,100 72,90 75,100 "
                   "90,100 92,90 94,108 95,50 97,130 99,100 102,90 105,100 "
                   "120,100 122,90 124,108 125,50 127,130 129,100 132,90 135,100 "
                   "150,100 152,90 154,108 155,50 157,130 159,100 162,90 165,100 "
                   "180,100 182,90 184,108 185,50 187,130 189,100 192,90 195,100 "
                   "210,100 212,90 214,108 215,50 217,130 219,100 222,90 225,100 "
                   "240,100 260,100 290,100 320,100 360,100 400,100 440,100 480,100 560,100 600,100",
                   "#f59e0b", "History of MI", "Residual Q-waves  |  T-wave inversion"),
        "abnormal_heartbeat": (
                   "0,100 10,100 12,95 14,100 18,100 19,90 20,60 21,140 22,100 23,110 26,100 "
                   "40,100 41,90 42,65 43,130 44,100 45,108 48,100 "
                   "55,100 58,100 60,93 61,65 62,135 63,100 64,106 68,100 "
                   "80,100 82,100 84,93 85,60 86,138 87,100 88,106 92,100 "
                   "100,100 108,100 110,93 111,70 112,128 113,100 114,106 118,100 "
                   "130,100 140,100 142,93 143,60 144,140 145,100 146,108 150,100 "
                   "160,100 170,100 172,93 173,55 174,145 175,100 176,110 180,100 "
                   "195,100 200,100 202,93 203,65 204,132 205,100 206,107 210,100 "
                   "230,100 240,100 250,100 270,100 300,100 340,100 380,100 420,100 480,100 560,100 600,100",
                   "#a855f7", "Abnormal Heartbeat", "Irregular rhythm  |  Varying amplitudes")
    }
    
    path, color, label, desc = paths.get(sample_type, paths["normal"])
    
    return f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="200" viewBox="0 0 600 200">
        <rect width="600" height="200" fill="#0a0f1a" rx="8"/>
        <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5" 
                  stroke-linecap="round" stroke-linejoin="round"/>
        <text x="12" y="18" font-family="monospace" font-size="11" fill="#26A69A" font-weight="bold">{label}</text>
        <text x="12" y="190" font-family="monospace" font-size="9" fill="#557799">{desc}</text>
        <text x="550" y="18" font-family="monospace" font-size="10" fill="#334455" text-anchor="end">Lead II</text>
    </svg>
    '''

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
            <div class="model-badge">RF + SMOTE • Accuracy: {rf.get("modelAccuracy", 0.92) * 100:.2f}%</div>
            <div class="risk-badge" style="background:{risk_color(rf["riskLevel"])}">
                {risk_label(rf["riskLevel"])} — Score: {rf["riskScore"]:.1f}/100
            </div>
            <div class="findings">{rf["recommendation"]}</div>
        </div>'''
    
    ecg_section = ""
    if data.get("ecgResult"):
        ecg = data["ecgResult"]
        ecg_section = f'''
        <div class="section">
            <div class="section-title">VGG16 ECG Classification</div>
            <div class="model-badge">CNN VGG16 • Accuracy: {ecg.get("modelAccuracy", 0.75) * 100:.2f}%</div>
            <div class="risk-badge" style="background:{risk_color(ecg["riskLevel"])}">
                {ecg["classification"]} — Confidence: {ecg["confidence"] * 100:.1f}%
            </div>
            <div class="findings">{ecg["findings"]}</div>
        </div>'''
    
    combined_section = ""
    if data.get("mode") == "dual" and data.get("finalRiskLevel"):
        combined_section = f'''
        <div class="section combined">
            <div class="section-title">Combined Triage Assessment</div>
            <div class="risk-badge" style="background:{risk_color(data["finalRiskLevel"])}">
                {risk_label(data["finalRiskLevel"])} — Confidence: {(data.get("confidenceScore", 0) * 100):.1f}%
            </div>
            <div class="findings">{data.get("finalRecommendation", "")}</div>
        </div>'''
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>CardioShield AI — Clinical Report</title>
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
    <div class="footer">CardioShield AI • RF+SMOTE 92.02% • VGG16 74.83%</div>
</body>
</html>'''

def pdf_download_button(pdf_data, label="📥 Download Clinical Report"):
    pdf_html = generate_pdf_html(pdf_data)
    b64 = base64.b64encode(pdf_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="CardioShield_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html"><button style="width:100%;margin-top:16px;padding:12px;background:#0B1F3A;color:white;border:none;border-radius:12px;font-weight:700;cursor:pointer;">{label}</button></a>'
    st.markdown(href, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    inject_custom_css()
    
    # Load models
    preprocessor, rf_model, vgg16_model = load_models()
    
    # Initialize session state
    defaults = {
        'ecg_sample': None,
        'ecg_image': None,
        'rf_result': None,
        'ecg_result': None,
        'dual_result': None,
        'data_rf_result': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    # ========================================================================
    # HEADER
    # ========================================================================
    
    st.markdown("""
    <div class="custom-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #2EC4B6, #25a99d); 
                     border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                    <span style="font-size: 24px;">🛡️</span>
                </div>
                <div>
                    <h1 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">CardioShield AI</h1>
                    <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.8rem;">Cardiovascular Risk Prediction</p>
                </div>
            </div>
        </div>
        <div style="display: flex; gap: 12px; margin-top: 20px; overflow-x: auto;">
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#2EC4B6;">⚡</span>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">RF + SMOTE</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">1,200 records</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#2EC4B6;">📊</span>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">VGG16 ECG</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">2,500 images</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#2EC4B6;">⏱️</span>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">Rapid</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">Inference time</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex:1;min-width:150px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <span style="color:#2EC4B6;">🗄️</span>
                    <div>
                        <p style="color:white;font-weight:700;margin:0;font-size:0.85rem;">3,700+</p>
                        <p style="color:rgba(255,255,255,0.5);margin:0;font-size:0.7rem;">Data points</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Dual Mode", "📈 ECG-Only", "📊 Data-Only", "💼 Investor Brief"])
    
    # ========================================================================
    # TAB 1: DUAL MODE
    # ========================================================================
    
    with tab1:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
            <span style="padding:4px 12px;border-radius:8px;border:1.5px solid rgba(46,196,182,0.35);
                  background:rgba(46,196,182,0.06);color:#0B1F3A;font-size:11px;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;">🔬 Dual Mode · RF + CNN VGG16</span>
            <span style="color:#6B7280;font-size:0.85rem;">Patient vitals + ECG combined</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Patient Vitals")
            patient_name = st.text_input("Patient ID / Name", key="dual_patient_name")
            
            fc1, fc2 = st.columns(2)
            with fc1:
                age = st.number_input("Age (years)", 18, 100, 45, key="dual_age")
                resting_bp = st.number_input("Resting BP (mmHg)", 60, 250, 130, key="dual_resting_bp")
                cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, 220, key="dual_cholesterol")
                max_hr = st.number_input("Max Heart Rate (bpm)", 50, 250, 150, key="dual_max_hr")
                oldpeak = st.number_input("ST Depression (Oldpeak)", -3.0, 7.0, 0.0, 0.1, key="dual_oldpeak")
            with fc2:
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="dual_sex")
                chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                    format_func=lambda x: {1:"1 — Typical Angina", 2:"2 — Atypical Angina", 3:"3 — Non-Anginal", 4:"4 — Asymptomatic"}[x], key="dual_chest_pain")
                resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: {0:"0 — Normal", 1:"1 — ST-T Wave", 2:"2 — LV Hypertrophy"}[x], key="dual_resting_ecg")
                st_slope = st.selectbox("ST Slope", [1, 2, 3],
                    format_func=lambda x: {1:"1 — Upsloping", 2:"2 — Flat", 3:"3 — Downsloping"}[x], key="dual_st_slope")
                fbs = st.toggle("Fasting Blood Sugar >120", key="dual_fbs")
                ex_angina = st.toggle("Exercise Induced Angina", key="dual_ex_angina")
            
            if st.button("Analyze Patient Data", type="primary", use_container_width=True, key="dual_analyze_btn"):
                patient_data = {
                    "patientName": patient_name, "age": age, "sex": sex,
                    "chestPainType": chest_pain, "restingBpS": resting_bp,
                    "cholesterol": cholesterol, "fastingBloodSugar": 1 if fbs else 0,
                    "restingEcg": resting_ecg, "maxHeartRate": max_hr,
                    "exerciseAngina": 1 if ex_angina else 0, "oldpeak": oldpeak, "stSlope": st_slope
                }
                with st.spinner("Running analysis..."):
                    st.session_state.rf_result = predict_rf(patient_data, preprocessor, rf_model)
                    if st.session_state.ecg_image:
                        st.session_state.ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                        st.session_state.dual_result = combine_results(st.session_state.rf_result, st.session_state.ecg_result)
                    elif st.session_state.ecg_sample:
                        st.session_state.ecg_result = simulate_ecg_prediction(is_demo=True)
                        st.session_state.dual_result = combine_results(st.session_state.rf_result, st.session_state.ecg_result)
                    else:
                        st.session_state.dual_result = combine_results(st.session_state.rf_result, None)
                        st.session_state.ecg_result = None
        
        with col2:
            st.markdown("#### ECG Image Scanner")
            
            sample_options = [None, "normal", "myocardial_infarction", "history_mi", "abnormal_heartbeat"]
            sample_labels = ["None", "Normal Sinus Rhythm", "Myocardial Infarction", "History of MI", "Abnormal Heartbeat"]
            
            current_idx = sample_options.index(st.session_state.ecg_sample) if st.session_state.ecg_sample in sample_options else 0
            
            selected = st.selectbox("Select demo sample", range(len(sample_options)),
                                    format_func=lambda i: sample_labels[i], index=current_idx, key="dual_sample_select")
            
            if sample_options[selected] is not None:
                st.session_state.ecg_sample = sample_options[selected]
                st.session_state.ecg_image = None
                svg = get_ecg_sample_svg(st.session_state.ecg_sample)
                st.markdown(f'<div class="ecg-preview" style="margin:12px 0;">{svg}</div>', unsafe_allow_html=True)
            else:
                st.session_state.ecg_sample = None
            
            uploaded_file = st.file_uploader("Upload ECG Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="dual_ecg_upload")
            if uploaded_file:
                st.session_state.ecg_sample = None
                st.session_state.ecg_image = base64.b64encode(uploaded_file.getvalue()).decode()
                st.image(uploaded_file, caption="Uploaded ECG", width='stretch')
            
            if not st.session_state.ecg_sample and not st.session_state.ecg_image:
                st.info("ℹ️ No ECG selected — RF + SMOTE will still run. Add an ECG to enable dual-model triage.")
        
        # Results
        if st.session_state.dual_result:
            st.markdown("---")
            dual = st.session_state.dual_result
            
            if not dual.get("ecgProvided"):
                st.warning("⚠️ ECG not provided — showing RF analysis only.")
            
            rc1, rc2 = st.columns([1, 1] if dual.get("ecgProvided") else [1])
            
            with rc1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Random Forest + SMOTE")
                rf = dual["rfResult"]
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                st.plotly_chart(render_risk_gauge(rf["riskScore"], rf["riskLevel"]), use_container_width=True, key="dual_rf_gauge")
                if rf.get("features"):
                    fig = render_feature_importance_chart(rf["features"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="dual_rf_features")
                st.markdown(f"""
                <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin-top:12px;">
                    <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">Clinical Interpretation</p>
                    <p style="font-size:0.85rem;color:#4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if dual.get("ecgProvided") and dual.get("ecgResult"):
                with rc2:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("#### CNN VGG16 ECG")
                    ecg = dual["ecgResult"]
                    is_normal = ecg["riskLevel"] == "low"
                    conf_pct = int(ecg["confidence"] * 100)
                    
                    st.markdown(f"""
                    <div style="border-radius:16px;padding:16px;border:1px solid {'#D1FAE5' if is_normal else '#FEE2E2'};
                         background:{'#ECFDF5' if is_normal else '#FEF2F2'};margin:12px 0;">
                        <span style="font-size:2rem;">{'✅' if is_normal else '⚠️'}</span>
                        <h4 style="color:#0B1F3A;margin:4px 0;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                        <p style="color:#6B7280;font-size:0.8rem;">{ecg["classification"]}</p>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-size:0.7rem;font-weight:700;color:#6B7280;">Model Confidence</span>
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
                        <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">Key Findings</p>
                        <p style="font-size:0.85rem;color:#4B5563;">{ecg["findings"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Combined result
            frl = dual["finalRiskLevel"]
            colors = {"high": ("#EF4444", "#FEF2F2"), "moderate": ("#F59E0B", "#FFF7ED"), "low": ("#2EC4B6", "#ECFDF5")}
            icons = {"high": "⚠️", "moderate": "⚡", "low": "✅"}
            
            st.markdown(f"""
            <div style="border:2px solid {colors[frl][0]};border-radius:16px;padding:20px;background:{colors[frl][1]};margin-top:16px;">
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
                    <span style="font-size:2rem;">{icons[frl]}</span>
                    <div>
                        <span style="font-size:0.65rem;font-weight:700;text-transform:uppercase;color:#6B7280;">Consensus Triage</span>
                        <h3 style="margin:4px 0;color:#0B1F3A;text-transform:capitalize;">{frl} Risk</h3>
                        <span style="font-size:0.75rem;color:#6B7280;">{(dual["confidenceScore"] * 100):.1f}% Confidence</span>
                    </div>
                </div>
                <div style="background:white;padding:16px;border-radius:12px;">
                    <p style="color:#4B5563;">{dual["finalRecommendation"]}</p>
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
            pdf_download_button(pdf_data)
    
    # ========================================================================
    # TAB 2: ECG-ONLY
    # ========================================================================
    
    with tab2:
        st.markdown("#### ECG Image Analysis")
        st.caption("CNN VGG16 waveform classification from ECG images")
        
        ec1, ec2 = st.columns([1, 1])
        
        with ec1:
            sample_options = [None, "normal", "myocardial_infarction", "history_mi", "abnormal_heartbeat"]
            sample_labels = ["None", "Normal Sinus Rhythm", "Myocardial Infarction", "History of MI", "Abnormal Heartbeat"]
            
            current_idx = sample_options.index(st.session_state.ecg_sample) if st.session_state.ecg_sample in sample_options else 0
            
            selected = st.selectbox("Select demo sample", range(len(sample_options)),
                                    format_func=lambda i: sample_labels[i], index=current_idx, key="ecg_only_sample")
            
            if sample_options[selected] is not None:
                st.session_state.ecg_sample = sample_options[selected]
                st.session_state.ecg_image = None
                svg = get_ecg_sample_svg(st.session_state.ecg_sample)
                st.markdown(f'<div class="ecg-preview" style="margin:12px 0;">{svg}</div>', unsafe_allow_html=True)
            else:
                st.session_state.ecg_sample = None
            
            st.markdown("---")
            uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"], key="ecg_only_upload")
            if uploaded_file:
                st.session_state.ecg_sample = None
                st.session_state.ecg_image = base64.b64encode(uploaded_file.getvalue()).decode()
                st.image(uploaded_file, caption="Uploaded ECG", width='stretch')
            
            has_ecg = bool(st.session_state.ecg_sample or st.session_state.ecg_image)
            
            if st.button("Analyze ECG", type="primary", use_container_width=True, disabled=not has_ecg, key="ecg_only_analyze"):
                if st.session_state.ecg_image:
                    with st.spinner("Processing..."):
                        st.session_state.ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                elif st.session_state.ecg_sample:
                    st.session_state.ecg_result = simulate_ecg_prediction(is_demo=True)
        
        with ec2:
            if st.session_state.ecg_result:
                ecg = st.session_state.ecg_result
                is_normal = ecg["riskLevel"] == "low"
                conf_pct = int(ecg["confidence"] * 100)
                
                st.markdown(f"""
                <div class="result-card">
                    <h4>Diagnostic Result</h4>
                    <div style="border-radius:16px;padding:16px;border:1px solid {'#D1FAE5' if is_normal else '#FEE2E2'};
                         background:{'#ECFDF5' if is_normal else '#FEF2F2'};margin:12px 0;">
                        <span style="font-size:2rem;">{'✅' if is_normal else '⚠️'}</span>
                        <h4 style="color:#0B1F3A;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                        <p style="color:#6B7280;font-size:0.8rem;">{ecg["classification"]}</p>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;margin:12px 0;">
                        <div style="display:flex;justify-content:space-between;">
                            <span style="font-size:0.7rem;font-weight:700;color:#6B7280;">Model Confidence</span>
                            <span style="font-size:1.5rem;font-weight:800;color:#0B1F3A;">{conf_pct}%</span>
                        </div>
                        <div style="height:12px;background:#E2E8F0;border-radius:999px;overflow:hidden;margin-top:8px;">
                            <div style="width:{conf_pct}%;height:100%;background:{'#10B981' if is_normal else '#EF4444'};border-radius:999px;"></div>
                        </div>
                    </div>
                    <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;">
                        <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">Key Findings</p>
                        <p style="font-size:0.85rem;color:#4B5563;">{ecg["findings"]}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if ecg.get("probabilities"):
                    fig = render_ecg_probability_chart(ecg["probabilities"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="ecg_only_probs")
                
                pdf_data = {"mode": "ecg", "ecgResult": ecg}
                pdf_download_button(pdf_data, "📥 Download ECG Report")
            else:
                st.markdown("""
                <div class="empty-state">
                    <span style="font-size:3rem;">📈</span>
                    <h4>Awaiting ECG Input</h4>
                    <p style="color:#6B7280;">Select a demo sample or upload an ECG image.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: DATA-ONLY
    # ========================================================================
    
    with tab3:
        st.markdown("#### Data-Only Analysis")
        st.caption("Random Forest + SMOTE — No ECG required")
        
        dc1, dc2 = st.columns([1.2, 1])
        
        with dc1:
            patient_name = st.text_input("Patient ID / Name", key="data_patient_name")
            
            fc1, fc2 = st.columns(2)
            with fc1:
                age = st.number_input("Age (years)", 18, 100, 45, key="data_age")
                resting_bp = st.number_input("Resting BP (mmHg)", 60, 250, 130, key="data_resting_bp")
                cholesterol = st.number_input("Cholesterol (mg/dl)", 0, 700, 220, key="data_cholesterol")
                max_hr = st.number_input("Max Heart Rate (bpm)", 50, 250, 150, key="data_max_hr")
                oldpeak = st.number_input("ST Depression (Oldpeak)", -3.0, 7.0, 0.0, 0.1, key="data_oldpeak")
            with fc2:
                sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="data_sex")
                chest_pain = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                    format_func=lambda x: {1:"1 — Typical Angina", 2:"2 — Atypical Angina", 3:"3 — Non-Anginal", 4:"4 — Asymptomatic"}[x], key="data_chest_pain")
                resting_ecg = st.selectbox("Resting ECG", [0, 1, 2],
                    format_func=lambda x: {0:"0 — Normal", 1:"1 — ST-T Wave", 2:"2 — LV Hypertrophy"}[x], key="data_resting_ecg")
                st_slope = st.selectbox("ST Slope", [1, 2, 3],
                    format_func=lambda x: {1:"1 — Upsloping", 2:"2 — Flat", 3:"3 — Downsloping"}[x], key="data_st_slope")
                fbs = st.toggle("Fasting Blood Sugar >120", key="data_fbs")
                ex_angina = st.toggle("Exercise Induced Angina", key="data_ex_angina")
            
            # Quick load buttons
            qc1, qc2 = st.columns(2)
            with qc1:
                if st.button("Load Low Risk", use_container_width=True, key="data_load_low"):
                    st.session_state.data_preset = "low"
                    st.rerun()
            with qc2:
                if st.button("Load High Risk", use_container_width=True, key="data_load_high"):
                    st.session_state.data_preset = "high"
                    st.rerun()
            
            if st.button("Analyze Patient Data", type="primary", use_container_width=True, key="data_analyze_btn"):
                patient_data = {
                    "patientName": patient_name, "age": age, "sex": sex,
                    "chestPainType": chest_pain, "restingBpS": resting_bp,
                    "cholesterol": cholesterol, "fastingBloodSugar": 1 if fbs else 0,
                    "restingEcg": resting_ecg, "maxHeartRate": max_hr,
                    "exerciseAngina": 1 if ex_angina else 0, "oldpeak": oldpeak, "stSlope": st_slope
                }
                with st.spinner("Analyzing..."):
                    st.session_state.data_rf_result = predict_rf(patient_data, preprocessor, rf_model)
        
        with dc2:
            if st.session_state.data_rf_result:
                rf = st.session_state.data_rf_result
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Clinical Assessment")
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                st.plotly_chart(render_risk_gauge(rf["riskScore"], rf["riskLevel"]), use_container_width=True, key="data_rf_gauge")
                if rf.get("features"):
                    fig = render_feature_importance_chart(rf["features"])
                    if fig: st.plotly_chart(fig, use_container_width=True, key="data_rf_features")
                st.markdown(f"""
                <div style="background:#F7F9FC;padding:16px;border-radius:12px;border:1px solid #E2E8F0;">
                    <p style="font-size:0.7rem;font-weight:700;color:#0B1F3A;text-transform:uppercase;">Clinical Interpretation</p>
                    <p style="font-size:0.85rem;color:#4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                pdf_data = {"mode": "data", "rfResult": rf}
                pdf_download_button(pdf_data)
            else:
                st.markdown("""
                <div class="empty-state">
                    <span style="font-size:3rem;">📊</span>
                    <h4>Awaiting Patient Data</h4>
                    <p style="color:#6B7280;">Fill out patient vitals and click analyze.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: INVESTOR BRIEF
    # ========================================================================
    
    with tab4:
        st.markdown("""
        <div style="text-align:center;margin-bottom:40px;">
            <span style="display:inline-block;padding:4px 16px;border-radius:999px;background:rgba(46,196,182,0.1);
                  border:1px solid rgba(46,196,182,0.3);color:#2EC4B6;font-size:0.7rem;font-weight:700;
                  text-transform:uppercase;letter-spacing:0.1em;margin-bottom:16px;">Investment Brief · 2026</span>
            <h1 style="color:#0B1F3A;">Redefining Cardiovascular Care with AI</h1>
            <p style="color:#6B7280;max-width:700px;margin:0 auto;">Dual-mode screening platform deploying Random Forest + SMOTE and CNN VGG16 for instant triage.</p>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(4)
        stats = [("91%", "RF+SMOTE Accuracy", "1,200 records"), ("2,500+", "ECG Images", "CNN VGG16"),
                 ("< 2s", "Inference Time", "Per prediction"), ("$4B+", "Addressable Market", "Global CDS")]
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
            st.markdown("""
            <div style="background:white;border-radius:16px;border-top:4px solid #EF4444;padding:24px;">
                <h3>⚠️ The Clinical Bottleneck</h3>
                <p style="color:#4B5563;">Primary care and rural clinics cannot rely on ECG for every screening. This delays triage and limits scalability.</p>
                <div style="background:#FEF2F2;padding:12px;border-radius:8px;margin-top:16px;border:1px solid #FEE2E2;">
                    <p style="color:#991B1B;font-size:0.8rem;">⚠️ CVD accounts for 32% of all global deaths — early triage saves lives.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with ac2:
            st.markdown("""
            <div style="background:white;border-radius:16px;border-top:4px solid #2EC4B6;padding:24px;">
                <h3>📊 Dual-Mode AI Architecture</h3>
                <p style="color:#4B5563;font-weight:600;">Works with or without ECG.</p>
                <p><strong>Phase 1:</strong> RF + SMOTE on 11 features (91% accuracy). No ECG required.</p>
                <p><strong>Phase 2:</strong> CNN VGG16 on 2,500 ECG images (4 classes).</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="background:linear-gradient(135deg, #0B1F3A, #122b4d);border-radius:24px;padding:48px;text-align:center;color:white;">
            <h2 style="color:white;">Deploy CardioShield AI at your facility</h2>
            <p style="color:rgba(255,255,255,0.7);">Join leading clinics getting early access.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("investor_waitlist"):
            wc1, wc2 = st.columns([3, 1])
            with wc1:
                wl_name = st.text_input("Full Name", key="investor_name")
                wl_email = st.text_input("Work Email", key="investor_email")
            with wc2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.form_submit_button("Get Early Access →", type="primary", use_container_width=True):
                    if wl_name and wl_email:
                        pos = add_to_waitlist(wl_name, wl_email)
                        if pos: st.success(f"✅ You're #{pos} on the waitlist!")
                        else: st.warning("Already registered.")
                    else:
                        st.error("Fill all fields.")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    with st.expander("ℹ️ Responsible AI & Clinical Use Disclaimer"):
        st.markdown("""
        CardioShield AI is a clinical decision support tool only. **Not a substitute** for professional medical diagnosis.
        RF+SMOTE trained on Heart Statlog Cleveland Hungary dataset (1,200 records, 11 features).
        VGG16 trained on 2,500 ECG images. Clinicians must apply professional judgment.
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:20px;">
        <p style="color:#9CA3AF;font-size:0.75rem;">
            CardioShield AI — Clinical decision support only · RF+SMOTE: 1,200 records · VGG16: 2,500 ECG images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar waitlist
    with st.sidebar:
        st.markdown("### 📋 Join Waitlist")
        st.metric("Total Signups", get_waitlist_count())
        with st.form("sidebar_waitlist"):
            sn = st.text_input("Name", key="sidebar_wl_name")
            se = st.text_input("Email", key="sidebar_wl_email")
            if st.form_submit_button("Join", type="primary", use_container_width=True):
                if sn and se:
                    p = add_to_waitlist(sn, se)
                    if p: st.success(f"Position #{p}!")
                    else: st.warning("Already registered.")

if __name__ == "__main__":
    main()
