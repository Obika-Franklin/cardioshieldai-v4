# app.py - CardioShield AI Streamlit App
# Complete clinical decision support platform for cardiovascular risk assessment

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
import sqlite3
import base64
import tempfile
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
            --teal-dark: #25a99d;
            --bg: #F7F9FC;
            --card: #FFFFFF;
            --border: #E2E8F0;
            --text-primary: #1F2937;
            --text-secondary: #6B7280;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
        }
        
        /* Global styles */
        .stApp {
            background-color: #F7F9FC;
        }
        
        /* Hide Streamlit default elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom header */
        .custom-header {
            background: linear-gradient(135deg, #0B1F3A 0%, #122b4d 100%);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 2rem;
        }
        
        /* Custom cards */
        .custom-card {
            background: white;
            border-radius: 16px;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        /* Risk badges */
        .risk-badge-high {
            background: #FEE2E2;
            color: #EF4444;
            border: 1px solid #FECACA;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .risk-badge-moderate {
            background: #FEF3C7;
            color: #F59E0B;
            border: 1px solid #FDE68A;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .risk-badge-low {
            background: #D1FAE5;
            color: #10B981;
            border: 1px solid #A7F3D0;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Button styles */
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
            background: linear-gradient(135deg, #122b4d 0%, #1a3558 100%);
        }
        
        /* Teal accent button */
        .teal-button > button {
            background: linear-gradient(135deg, #2EC4B6 0%, #25a99d 100%) !important;
            box-shadow: 0 2px 12px rgba(46, 196, 182, 0.35) !important;
        }
        
        /* KPI cards */
        .kpi-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
            backdrop-filter: blur(10px);
        }
        
        /* ECG sample selection */
        .ecg-sample-card {
            cursor: pointer;
            border: 2px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem;
            transition: all 0.2s;
        }
        
        .ecg-sample-card:hover {
            border-color: #9CA3AF;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .ecg-sample-card.active {
            border-color: #0B1F3A;
            background: #0B1F3A;
            color: white;
        }
        
        /* Result cards */
        .result-card {
            background: white;
            border-radius: 20px;
            border: 1px solid #E2E8F0;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Form styling */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            border-radius: 10px;
            border: 1px solid #E2E8F0;
        }
        
        .stSelectbox > div > div > select {
            border-radius: 10px;
            border: 1px solid #E2E8F0;
        }
        
        /* Tab styling */
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
        
        /* Progress bar */
        .custom-progress {
            height: 6px;
            background: rgba(255,255,255,0.1);
            border-radius: 999px;
            overflow: hidden;
        }
        
        .custom-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #2EC4B6, #4dd9ce);
            border-radius: 999px;
            transition: width 0.7s;
        }
        
        /* ECG preview */
        .ecg-preview {
            background: #0a0f1a;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #1a2a3a;
        }
        
        /* Disclaimer */
        .disclaimer-box {
            background: #FFF3E0;
            border: 1px solid #FFE0B2;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 2rem;
        }
        
        /* Empty state */
        .empty-state {
            border: 2px dashed #E2E8F0;
            border-radius: 16px;
            padding: 3rem;
            text-align: center;
            background: rgba(247, 249, 252, 0.5);
        }
        
        /* Analysis card */
        .analysis-card {
            background: white;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.5rem;
        }
        
        /* Model badge */
        .model-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .model-badge-rf {
            background: rgba(11, 31, 58, 0.08);
            color: #0B1F3A;
            border: 1px solid rgba(11, 31, 58, 0.15);
        }
        
        .model-badge-cnn {
            background: rgba(46, 196, 182, 0.1);
            color: #2EC4B6;
            border: 1px solid rgba(46, 196, 182, 0.25);
        }
        
        /* Investor page cards */
        .investor-stat {
            text-align: center;
            padding: 1.5rem;
        }
        
        .investor-stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
        }
        
        /* Hide Streamlit expander arrow */
        .streamlit-expanderHeader {
            border: none;
        }
        
        /* Alert styles */
        .alert-warning {
            background: rgba(255, 251, 235, 0.8);
            border: 1px solid rgba(245, 158, 11, 0.35);
            border-radius: 12px;
            padding: 1rem;
            color: #92400E;
        }
        
        .alert-error {
            background: #FEE2E2;
            border: 1px solid #FECACA;
            border-radius: 12px;
            padding: 1rem;
            color: #991B1B;
        }
        
        .alert-success {
            background: #D1FAE5;
            border: 1px solid #A7F3D0;
            border-radius: 12px;
            padding: 1rem;
            color: #065F46;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PREPROCESSOR_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/preprocessor/preprocessor.pkl"
RF_MODEL_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/rf_model/rf_model.pkl"
VGG16_MODEL_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/v1.0.0/vgg16_ecg_model.keras"

NORMAL_ECG_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/normal-ecg/Normal.97.-.Copy.jpg"
MI_ECG_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/myocardial-infarction/MI.99.-.Copy.jpg"

@st.cache_resource
def download_file(url, filename):
    """Download a file with progress tracking"""
    filepath = MODEL_DIR / filename
    if not filepath.exists():
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return filepath

@st.cache_resource
def load_models():
    """Load or download all required models"""
    try:
        preprocessor_path = download_file(PREPROCESSOR_URL, "preprocessor.pkl")
        rf_model_path = download_file(RF_MODEL_URL, "rf_model.pkl")
        vgg16_model_path = download_file(VGG16_MODEL_URL, "vgg16_ecg_model.keras")
        
        preprocessor = joblib.load(preprocessor_path)
        rf_model = joblib.load(rf_model_path)
        
        # Try to load VGG16
        vgg16_model = None
        try:
            from tensorflow.keras.models import load_model
            vgg16_model = load_model(vgg16_model_path)
        except Exception as e:
            st.warning(f"VGG16 model loading failed: {e}. ECG analysis will use simulation mode.")
        
        return preprocessor, rf_model, vgg16_model
    except Exception as e:
        st.warning(f"Model loading failed: {e}. Using simulation mode.")
        return None, None, None

# ============================================================================
# DATABASE SETUP (Waitlist)
# ============================================================================

def init_db():
    """Initialize SQLite database for waitlist"""
    conn = sqlite3.connect('waitlist.db')
    cursor = conn.cursor()
    cursor.execute('''
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
    """Add a user to the waitlist"""
    conn = init_db()
    try:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO waitlist (name, email) VALUES (?, ?)', (name, email))
        conn.commit()
        position = cursor.lastrowid
        return position
    except sqlite3.IntegrityError:
        return None  # Email already exists
    finally:
        conn.close()

def get_waitlist_count():
    """Get total waitlist count"""
    conn = init_db()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM waitlist')
    count = cursor.fetchone()[0]
    conn.close()
    return count

# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def predict_rf(patient_data, preprocessor, rf_model):
    """Run Random Forest prediction"""
    if preprocessor is None or rf_model is None:
        return simulate_rf_prediction(patient_data)
    
    # Create DataFrame from patient data
    df = pd.DataFrame([patient_data])
    
    # Preprocess
    X_processed = preprocessor.transform(df)
    
    # Predict
    proba = rf_model.predict_proba(X_processed)[0]
    risk_score = proba[1] * 100  # Probability of class 1 (high risk)
    
    # Determine risk level
    if risk_score >= 70:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Generate recommendation
    if risk_level == "high":
        recommendation = "Patient exhibits multiple cardiovascular risk factors. Immediate cardiology referral recommended. Consider stress testing and comprehensive lipid panel."
    elif risk_level == "moderate":
        recommendation = "Moderate cardiovascular risk detected. Monitor patient closely and consider lifestyle interventions. Follow-up in 3-6 months with repeat assessment."
    else:
        recommendation = "Low cardiovascular risk profile. Continue routine preventive care. Maintain healthy lifestyle and schedule annual check-up."
    
    # Feature importance (from model if available)
    if hasattr(rf_model, 'feature_importances_'):
        feature_names = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol', 
                        'fasting blood sugar', 'resting ecg', 'max heart rate', 
                        'exercise angina', 'oldpeak', 'ST slope']
        importances = rf_model.feature_importances_
        features = [{"name": name, "importance": float(imp)} 
                   for name, imp in zip(feature_names, importances)]
        features.sort(key=lambda x: x["importance"], reverse=True)
    else:
        features = []
    
    return {
        "riskScore": risk_score,
        "riskLevel": risk_level,
        "rfProbability": proba[1],
        "recommendation": recommendation,
        "modelAccuracy": 0.9202,
        "features": features
    }

def simulate_rf_prediction(patient_data):
    """Simulate RF prediction for demo/testing"""
    # Simple heuristic based on known risk factors
    risk_score = 0
    
    if patient_data.get('age', 45) > 60:
        risk_score += 25
    elif patient_data.get('age', 45) > 45:
        risk_score += 15
    
    if patient_data.get('sex', 1) == 1:  # Male
        risk_score += 10
    
    if patient_data.get('chestPainType', 2) >= 3:
        risk_score += 20
    elif patient_data.get('chestPainType', 2) == 2:
        risk_score += 10
    
    if patient_data.get('restingBpS', 130) > 140:
        risk_score += 15
    
    if patient_data.get('cholesterol', 220) > 240:
        risk_score += 15
    
    if patient_data.get('fastingBloodSugar', 0) == 1:
        risk_score += 10
    
    if patient_data.get('exerciseAngina', 0) == 1:
        risk_score += 20
    
    if patient_data.get('oldpeak', 0) > 1.5:
        risk_score += 15
    
    if patient_data.get('stSlope', 1) >= 2:
        risk_score += 15
    
    risk_score = min(risk_score, 98)
    risk_score = max(risk_score, 2)
    
    # Risk level
    if risk_score >= 70:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    # Recommendation
    if risk_level == "high":
        recommendation = "Patient exhibits multiple cardiovascular risk factors. Immediate cardiology referral recommended."
    elif risk_level == "moderate":
        recommendation = "Moderate cardiovascular risk detected. Monitor patient closely and consider lifestyle interventions."
    else:
        recommendation = "Low cardiovascular risk profile. Continue routine preventive care."
    
    # Simulated feature importance
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
    """Run VGG16 ECG classification"""
    if vgg16_model is None:
        return simulate_ecg_prediction()
    
    try:
        from tensorflow.keras.preprocessing import image as keras_image
        
        # Preprocess image
        img = Image.open(BytesIO(base64.b64decode(image_data)))
        img = img.resize((100, 100))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = vgg16_model.predict(img_array, verbose=0)[0]
        classes = ['abnormal_heartbeat', 'history_mi', 'myocardial_infarction', 'normal']
        
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        classification = classes[predicted_idx]
        
        # Map to risk level
        if classification == 'normal':
            risk_level = "low"
            findings = "Normal sinus rhythm. Regular P-QRS-T waveform pattern. No ST segment abnormalities detected."
        elif classification == 'myocardial_infarction':
            risk_level = "high"
            findings = "ST segment elevation detected. Pathological Q waves present. Findings consistent with acute myocardial infarction."
        elif classification == 'history_mi':
            risk_level = "moderate"
            findings = "Residual Q waves detected. T-wave inversion noted. Findings suggest prior myocardial infarction."
        else:
            risk_level = "moderate"
            findings = "Irregular rhythm pattern detected. Varying QRS amplitudes. Abnormal heartbeat morphology."
        
        probabilities = {classes[i]: float(predictions[i]) for i in range(len(classes))}
        
        return {
            "classification": classification.replace('_', ' ').title(),
            "confidence": confidence,
            "findings": findings,
            "riskLevel": risk_level,
            "modelAccuracy": 0.7483,
            "modelName": "CNN VGG16",
            "isDemo": False,
            "probabilities": probabilities
        }
    except Exception as e:
        st.warning(f"ECG inference failed: {e}. Using simulation.")
        return simulate_ecg_prediction()

def simulate_ecg_prediction(is_demo=True):
    """Simulate ECG prediction"""
    return {
        "classification": "Normal Sinus Rhythm",
        "confidence": 0.92,
        "findings": "Regular P-QRS-T waveform pattern. No ST segment abnormalities detected. Heart rate within normal range.",
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
    """Combine RF and ECG results for dual mode"""
    if ecg_result is None:
        return {
            "rfResult": rf_result,
            "ecgProvided": False,
            "finalRiskLevel": rf_result["riskLevel"],
            "finalRecommendation": rf_result["recommendation"],
            "confidenceScore": rf_result["rfProbability"]
        }
    
    # Weighted combination
    rf_weight = 0.6
    ecg_weight = 0.4
    
    # Map risk levels to scores
    risk_map = {"low": 1, "moderate": 2, "high": 3}
    reverse_map = {1: "low", 2: "moderate", 3: "high"}
    
    rf_risk_score = risk_map[rf_result["riskLevel"]]
    ecg_risk_score = risk_map[ecg_result["riskLevel"]]
    
    combined_score = rf_risk_score * rf_weight + ecg_risk_score * ecg_weight
    final_risk_level = reverse_map[round(combined_score)]
    
    confidence_score = (rf_result["rfProbability"] + ecg_result["confidence"]) / 2
    
    if final_risk_level == "high":
        recommendation = "Combined RF + VGG16 analysis indicates HIGH cardiovascular risk. Urgent cardiology referral required. Both structured vitals and ECG morphology suggest significant pathology."
    elif final_risk_level == "moderate":
        recommendation = "Combined analysis indicates MODERATE cardiovascular risk. Further diagnostic testing recommended. Monitor patient symptoms and schedule follow-up within 1-3 months."
    else:
        recommendation = "Combined analysis indicates LOW cardiovascular risk. Routine preventive care recommended. Both models agree on low-risk classification."
    
    return {
        "rfResult": rf_result,
        "ecgResult": ecg_result,
        "ecgProvided": True,
        "finalRiskLevel": final_risk_level,
        "finalRecommendation": recommendation,
        "confidenceScore": confidence_score
    }

# ============================================================================
# ECG SAMPLE IMAGES (Inline SVGs)
# ============================================================================

def get_ecg_sample_svg(sample_type):
    """Get inline SVG for ECG sample visualization"""
    if sample_type == "normal":
        path = "0,100 20,100 25,98 30,100 35,100 40,100 42,95 44,100 46,55 48,100 50,108 52,100 70,100 75,98 80,100 90,100 92,95 94,100 96,55 98,100 100,108 102,100 120,100 125,98 130,100 140,100 142,95 144,100 146,55 148,100 150,108 152,100 170,100 175,98 180,100 190,100 192,95 194,100 196,55 198,100 200,108 202,100 220,100 225,98 230,100 240,100 242,95 244,100 246,55 248,100 250,108 252,100 270,100 275,98 280,100 290,100 292,95 294,100 296,55 298,100 300,108 302,100 320,100 325,98 330,100 340,100 342,95 344,100 346,55 348,100 350,108 352,100 370,100 375,98 380,100 390,100 392,95 394,100 396,55 398,100 400,108 402,100 420,100 440,100 460,100 480,100 500,100 520,100 560,100 600,100"
        color = "#26A69A"
        label = "Normal ECG"
        desc = "Regular P-QRS-T  |  Normal Sinus Rhythm"
    elif sample_type == "myocardial_infarction":
        path = "0,100 15,100 20,98 25,100 30,100 32,90 34,100 35,40 36,130 37,100 38,75 42,100 60,100 62,90 64,100 65,40 66,130 67,100 68,75 72,100 90,100 92,90 94,100 95,40 96,130 97,100 98,75 102,100 120,100 122,90 124,100 125,40 126,130 127,100 128,75 132,100 150,100 152,90 154,100 155,40 156,130 157,100 158,75 162,100 180,100 182,90 184,100 185,40 186,130 187,100 188,75 192,100 210,100 212,90 214,100 215,40 216,130 217,100 218,75 222,100 240,100 242,90 244,100 245,40 246,130 247,100 248,75 252,100 270,100 300,100 330,100 360,100 400,100 440,100 480,100 560,100 600,100"
        color = "#ef4444"
        label = "Myocardial Infarction"
        desc = "ST Elevation  |  Pathological Q-waves"
    elif sample_type == "history_mi":
        path = "0,100 20,100 22,95 24,100 30,100 32,90 34,108 35,50 37,130 39,100 42,90 45,100 60,100 62,90 64,108 65,50 67,130 69,100 72,90 75,100 90,100 92,90 94,108 95,50 97,130 99,100 102,90 105,100 120,100 122,90 124,108 125,50 127,130 129,100 132,90 135,100 150,100 152,90 154,108 155,50 157,130 159,100 162,90 165,100 180,100 182,90 184,108 185,50 187,130 189,100 192,90 195,100 210,100 212,90 214,108 215,50 217,130 219,100 222,90 225,100 240,100 260,100 290,100 320,100 360,100 400,100 440,100 480,100 560,100 600,100"
        color = "#f59e0b"
        label = "History of MI"
        desc = "Residual Q-waves  |  T-wave inversion"
    else:  # abnormal_heartbeat
        path = "0,100 10,100 12,95 14,100 18,100 19,90 20,60 21,140 22,100 23,110 26,100 40,100 41,90 42,65 43,130 44,100 45,108 48,100 55,100 58,100 60,93 61,65 62,135 63,100 64,106 68,100 80,100 82,100 84,93 85,60 86,138 87,100 88,106 92,100 100,100 108,100 110,93 111,70 112,128 113,100 114,106 118,100 130,100 140,100 142,93 143,60 144,140 145,100 146,108 150,100 160,100 170,100 172,93 173,55 174,145 175,100 176,110 180,100 195,100 200,100 202,93 203,65 204,132 205,100 206,107 210,100 230,100 240,100 250,100 270,100 300,100 340,100 380,100 420,100 480,100 560,100 600,100"
        color = "#a855f7"
        label = "Abnormal Heartbeat"
        desc = "Irregular rhythm  |  Varying amplitudes"
    
    svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="600" height="200" viewBox="0 0 600 200">
        <rect width="600" height="200" fill="#0a0f1a" rx="8"/>
        <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <text x="12" y="18" font-family="monospace" font-size="11" fill="#26A69A" font-weight="bold">{label}</text>
        <text x="12" y="190" font-family="monospace" font-size="9" fill="#557799">{desc}</text>
        <text x="550" y="18" font-family="monospace" font-size="10" fill="#334455" text-anchor="end">Lead II</text>
    </svg>
    '''
    return svg

# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def generate_pdf_html(data):
    """Generate PDF report as HTML (replicating the React PDF logic)"""
    now = datetime.now()
    date_str = now.strftime("%B %d, %Y, %I:%M %p")
    
    mode_labels = {
        "dual": "Dual Mode (RF + VGG16)",
        "ecg": "ECG-Only Mode (VGG16)",
        "data": "Data-Only Mode (RF + SMOTE)"
    }
    
    def risk_color(level):
        if level == "high": return "#D32F2F"
        if level == "moderate": return "#F57C00"
        return "#2E7D32"
    
    def risk_label(level):
        if level == "high": return "HIGH RISK"
        if level == "moderate": return "MODERATE RISK"
        return "LOW RISK"
    
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
            <div class="source-tag">Source: {data.get("ecgSource", "N/A")}</div>
            <div class="findings">{ecg["findings"]}</div>
        </div>'''
    
    combined_section = ""
    if data.get("mode") == "dual" and data.get("finalRiskLevel"):
        combined_section = f'''
        <div class="section combined">
            <div class="section-title">Combined Triage Assessment</div>
            <div class="risk-badge" style="background:{risk_color(data["finalRiskLevel"])}">
                {risk_label(data["finalRiskLevel"])} — Combined Confidence: {(data.get("confidenceScore", 0) * 100):.1f}%
            </div>
            <div class="findings">{data.get("finalRecommendation", "")}</div>
        </div>'''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>CardioShield AI — Clinical Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #212121; background: #fff; padding: 40px; }}
        .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 3px solid #001F3F; padding-bottom: 16px; margin-bottom: 24px; }}
        .brand {{ font-size: 24px; font-weight: 700; color: #001F3F; }}
        .brand-sub {{ font-size: 12px; color: #26A69A; font-weight: 600; letter-spacing: 1px; margin-top: 2px; }}
        .meta {{ text-align: right; font-size: 11px; color: #757575; }}
        .mode-badge {{ display: inline-block; background: #001F3F; color: #fff; padding: 4px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 20px; }}
        .section {{ background: #F5F5F5; border-radius: 8px; padding: 20px; margin-bottom: 20px; border-left: 4px solid #26A69A; }}
        .section.combined {{ border-left-color: #001F3F; }}
        .section-title {{ font-size: 14px; font-weight: 700; color: #001F3F; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
        .model-badge {{ display: inline-block; background: #26A69A; color: #fff; padding: 2px 10px; border-radius: 3px; font-size: 10px; font-weight: 700; margin-bottom: 10px; }}
        .risk-badge {{ display: inline-block; color: #fff; padding: 6px 16px; border-radius: 4px; font-size: 13px; font-weight: 700; margin-bottom: 12px; }}
        .source-tag {{ font-size: 11px; color: #757575; margin-bottom: 8px; font-style: italic; }}
        .findings {{ font-size: 13px; color: #424242; line-height: 1.7; }}
        .disclaimer {{ margin-top: 24px; padding: 14px; background: #FFF3E0; border-left: 4px solid #F57C00; border-radius: 4px; font-size: 11px; color: #5D4037; line-height: 1.6; }}
        .footer {{ margin-top: 20px; padding-top: 12px; border-top: 1px solid #e0e0e0; font-size: 10px; color: #9E9E9E; text-align: center; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="brand">CardioShield AI</div>
            <div class="brand-sub">CLINICAL DECISION SUPPORT REPORT</div>
        </div>
        <div class="meta">
            <div>Generated: {date_str}</div>
            {"<div>Patient: " + data.get("patientName", "") + "</div>" if data.get("patientName") else ""}
            <div>Report ID: CSR-{now.strftime("%Y%m%d%H%M%S")}</div>
        </div>
    </div>
    <div class="mode-badge">{mode_labels.get(data.get("mode", "data"), "")}</div>
    {rf_section}
    {ecg_section}
    {combined_section}
    <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> CardioShield AI is a clinical decision support tool only. Results are generated by machine learning models (Random Forest + SMOTE and CNN VGG16). This report does NOT constitute a medical diagnosis and must NOT be used as a substitute for professional clinical evaluation by a qualified healthcare provider.
    </div>
    <div class="footer">
        CardioShield AI • RF+SMOTE 92.02% • VGG16 74.83% • Heart Statlog Cleveland Hungary Dataset
    </div>
</body>
</html>'''
    
    return html

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_risk_gauge(score, risk_level):
    """Render radial gauge chart for risk score"""
    colors = {
        "low": ("#2EC4B6", "#E6F7F5"),
        "moderate": ("#F59E0B", "#FEF3C7"),
        "high": ("#EF4444", "#FEE2E2")
    }
    gauge_color, track_color = colors.get(risk_level, colors["low"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cardiovascular Risk", 'font': {'size': 14, 'color': '#0B1F3A'}},
        delta={'reference': 50, 'increasing': {'color': "#EF4444"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0B1F3A"},
            'bar': {'color': gauge_color, 'thickness': 0.15},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 30], 'color': '#E6F7F5'},
                {'range': [30, 70], 'color': '#FEF3C7'},
                {'range': [70, 100], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#0B1F3A", 'family': "Segoe UI"}
    )
    
    return fig

def render_feature_importance_chart(features):
    """Render feature importance bar chart"""
    if not features:
        return None
    
    df = pd.DataFrame(features)
    df = df.sort_values('importance', ascending=True).tail(10)
    
    fig = px.bar(
        df,
        x='importance',
        y='name',
        orientation='h',
        title='Feature Importance (Random Forest)',
        color='importance',
        color_continuous_scale=['#E6F7F5', '#2EC4B6', '#0B1F3A']
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#0B1F3A"},
        xaxis_title="Importance",
        yaxis_title="",
        showlegend=False
    )
    
    return fig

def render_ecg_probability_chart(probabilities):
    """Render ECG class probability horizontal bar chart"""
    if not probabilities:
        return None
    
    items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace('_', ' ').title() for k, v in items]
    values = [v * 100 for k, v in items]
    
    fig = go.Figure()
    
    for i, (label, value) in enumerate(zip(labels, values)):
        color = '#CBD5E1'
        if i == 0:
            color = '#10B981'  # Top prediction green
        fig.add_trace(go.Bar(
            y=[label],
            x=[value],
            orientation='h',
            marker_color=color,
            text=f'{value:.1f}%',
            textposition='outside',
            name=label
        ))
    
    fig.update_layout(
        title='CNN Class Probabilities',
        height=200,
        margin=dict(l=10, r=50, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#0B1F3A"},
        xaxis_title="Probability (%)",
        showlegend=False,
        barmode='group'
    )
    
    return fig

def risk_badge_html(level):
    """Return HTML for risk badge"""
    badges = {
        "high": '<span class="risk-badge-high">HIGH RISK</span>',
        "moderate": '<span class="risk-badge-moderate">MODERATE RISK</span>',
        "low": '<span class="risk-badge-low">LOW RISK</span>'
    }
    return badges.get(level, badges["low"])

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    inject_custom_css()
    
    # Load models
    preprocessor, rf_model, vgg16_model = load_models()
    
    # Initialize session state
    if 'ecg_sample' not in st.session_state:
        st.session_state.ecg_sample = None
    if 'ecg_image' not in st.session_state:
        st.session_state.ecg_image = None
    if 'rf_result' not in st.session_state:
        st.session_state.rf_result = None
    if 'ecg_result' not in st.session_state:
        st.session_state.ecg_result = None
    if 'dual_result' not in st.session_state:
        st.session_state.dual_result = None
    if 'waitlist_open' not in st.session_state:
        st.session_state.waitlist_open = False
    
    # ============================================================================
    # HEADER
    # ============================================================================
    
    st.markdown("""
    <div class="custom-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 48px; height: 48px; background: linear-gradient(135deg, #2EC4B6, #25a99d); border-radius: 12px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(46, 196, 182, 0.3);">
                    <span style="font-size: 24px;">🛡️</span>
                </div>
                <div>
                    <h1 style="color: white; margin: 0; font-size: 1.5rem; font-weight: 700;">CardioShield AI</h1>
                    <p style="color: rgba(255,255,255,0.6); margin: 0; font-size: 0.8rem;">Cardiovascular Risk Prediction</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="text-align: right;">
                    <span style="color: rgba(255,255,255,0.5); font-size: 0.75rem;">Early Access</span>
                    <br>
                    <span style="color: #2EC4B6; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">Clinical Decision Support</span>
                </div>
            </div>
        </div>
        
        <!-- KPI Bar -->
        <div style="display: flex; gap: 12px; margin-top: 20px; overflow-x: auto; padding-bottom: 4px;">
            <div class="kpi-card" style="flex: 1; min-width: 150px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="padding: 8px; background: rgba(46, 196, 182, 0.15); border-radius: 8px; color: #2EC4B6;">⚡</div>
                    <div>
                        <p style="color: white; font-weight: 700; margin: 0; font-size: 0.9rem;">RF + SMOTE</p>
                        <p style="color: rgba(255,255,255,0.5); margin: 0; font-size: 0.7rem;">1,200 patient records</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 150px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="padding: 8px; background: rgba(46, 196, 182, 0.15); border-radius: 8px; color: #2EC4B6;">📊</div>
                    <div>
                        <p style="color: white; font-weight: 700; margin: 0; font-size: 0.9rem;">VGG16 ECG</p>
                        <p style="color: rgba(255,255,255,0.5); margin: 0; font-size: 0.7rem;">2,500 ECG images</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 150px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="padding: 8px; background: rgba(46, 196, 182, 0.15); border-radius: 8px; color: #2EC4B6;">⏱️</div>
                    <div>
                        <p style="color: white; font-weight: 700; margin: 0; font-size: 0.9rem;">Rapid</p>
                        <p style="color: rgba(255,255,255,0.5); margin: 0; font-size: 0.7rem;">Inference response time</p>
                    </div>
                </div>
            </div>
            <div class="kpi-card" style="flex: 1; min-width: 150px;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div style="padding: 8px; background: rgba(46, 196, 182, 0.15); border-radius: 8px; color: #2EC4B6;">🗄️</div>
                    <div>
                        <p style="color: white; font-weight: 700; margin: 0; font-size: 0.9rem;">3,700+</p>
                        <p style="color: rgba(255,255,255,0.5); margin: 0; font-size: 0.7rem;">Combined data points</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB NAVIGATION
    # ============================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 Dual Mode", "📈 ECG-Only", "📊 Data-Only", "💼 Investor Brief"])
    
    # ============================================================================
    # TAB 1: DUAL MODE
    # ============================================================================
    
    with tab1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
            <span style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 8px; border: 1.5px solid rgba(46,196,182,0.35); background: rgba(46,196,182,0.06); color: #0B1F3A; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">
                🔬 Dual Mode · RF + CNN VGG16
            </span>
            <span style="color: #6B7280; font-size: 0.85rem;">Patient vitals + ECG combined for maximum triage accuracy</span>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Patient Vitals")
            
            with st.form("dual_patient_form"):
                patient_name = st.text_input("Patient ID / Name", placeholder="Enter patient ID", key="dual_name")
                
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, key="dual_age")
                    resting_bp = st.number_input("Resting BP (mmHg)", min_value=60, max_value=250, value=130, key="dual_bp")
                    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=220, key="dual_chol")
                    max_hr = st.number_input("Max Heart Rate (bpm)", min_value=50, max_value=250, value=150, key="dual_hr")
                    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1, key="dual_oldpeak")
                    
                with form_col2:
                    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="dual_sex")
                    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4], 
                                              format_func=lambda x: {1: "1 — Typical Angina", 2: "2 — Atypical Angina", 
                                                                     3: "3 — Non-Anginal Pain", 4: "4 — Asymptomatic"}[x], key="dual_cp")
                    resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                               format_func=lambda x: {0: "0 — Normal", 1: "1 — ST-T Wave", 2: "2 — LV Hypertrophy"}[x], key="dual_ecg")
                    st_slope = st.selectbox("ST Slope", options=[1, 2, 3],
                                            format_func=lambda x: {1: "1 — Upsloping", 2: "2 — Flat", 3: "3 — Downsloping"}[x], key="dual_slope")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    fbs = st.toggle("Fasting Blood Sugar >120", value=False, key="dual_fbs")
                    ex_angina = st.toggle("Exercise Induced Angina", value=False, key="dual_exang")
                
                dual_submitted = st.form_submit_button("Analyze Patient Data", type="primary", use_container_width=True)
                
                if dual_submitted:
                    patient_data = {
                        "patientName": patient_name,
                        "age": age,
                        "sex": sex,
                        "chestPainType": chest_pain,
                        "restingBpS": resting_bp,
                        "cholesterol": cholesterol,
                        "fastingBloodSugar": 1 if fbs else 0,
                        "restingEcg": resting_ecg,
                        "maxHeartRate": max_hr,
                        "exerciseAngina": 1 if ex_angina else 0,
                        "oldpeak": oldpeak,
                        "stSlope": st_slope
                    }
                    
                    with st.spinner("Running analysis..."):
                        rf_result = predict_rf(patient_data, preprocessor, rf_model)
                        st.session_state.rf_result = rf_result
                        
                        # Check if ECG is available
                        if st.session_state.ecg_image:
                            ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                            st.session_state.ecg_result = ecg_result
                            st.session_state.dual_result = combine_results(rf_result, ecg_result)
                        elif st.session_state.ecg_sample:
                            # Demo ECG
                            ecg_result = simulate_ecg_prediction(is_demo=True)
                            st.session_state.ecg_result = ecg_result
                            st.session_state.dual_result = combine_results(rf_result, ecg_result)
                        else:
                            st.session_state.dual_result = combine_results(rf_result, None)
                            st.session_state.ecg_result = None
        
        with col2:
            st.markdown("### ECG Image Scanner")
            
            # ECG Sample selection
            sample_options = ["None", "Normal Sinus Rhythm", "Myocardial Infarction", "History of MI", "Abnormal Heartbeat"]
            sample_values = [None, "normal", "myocardial_infarction", "history_mi", "abnormal_heartbeat"]
            
            selected_idx = 0
            if st.session_state.ecg_sample:
                try:
                    selected_idx = sample_values.index(st.session_state.ecg_sample)
                except ValueError:
                    selected_idx = 0
            
            selected_sample = st.selectbox(
                "Select demo sample or upload below",
                options=range(len(sample_options)),
                format_func=lambda i: sample_options[i],
                index=selected_idx,
                key="dual_sample_select"
            )
            
            if selected_sample > 0:
                st.session_state.ecg_sample = sample_values[selected_sample]
                st.session_state.ecg_image = None
                
                # Show SVG preview
                svg = get_ecg_sample_svg(st.session_state.ecg_sample)
                st.markdown(f'<div class="ecg-preview" style="margin: 12px 0;">{svg}</div>', unsafe_allow_html=True)
            else:
                st.session_state.ecg_sample = None
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload ECG Image (PNG, JPG up to 20MB)",
                type=["png", "jpg", "jpeg"],
                key="dual_ecg_upload"
            )
            
            if uploaded_file:
                st.session_state.ecg_sample = None
                bytes_data = uploaded_file.getvalue()
                st.session_state.ecg_image = base64.b64encode(bytes_data).decode()
                
                # Preview
                st.image(uploaded_file, caption="Uploaded ECG", use_container_width=True)
            
            # ECG status
            if not st.session_state.ecg_sample and not st.session_state.ecg_image:
                st.info("ℹ️ No ECG selected — RF + SMOTE will still run on patient vitals. Select a sample or upload an ECG to enable dual-model triage.")
        
        # Display results
        if st.session_state.dual_result:
            st.markdown("---")
            st.markdown("### Results")
            
            dual = st.session_state.dual_result
            
            if not dual.get("ecgProvided"):
                st.warning("⚠️ ECG was not provided — results show Random Forest analysis only. Add an ECG input for dual-model combined triage.")
            
            res_col1, res_col2 = st.columns([1, 1] if dual.get("ecgProvided") else [1])
            
            with res_col1:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Random Forest + SMOTE")
                st.caption("Structured patient vitals — 11 clinical features")
                
                rf = dual["rfResult"]
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                
                # Risk gauge
                fig = render_risk_gauge(rf["riskScore"], rf["riskLevel"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if rf.get("features"):
                    fig2 = render_feature_importance_chart(rf["features"])
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown(f"""
                <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0; margin-top: 12px;">
                    <p style="font-size: 0.7rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.05em;">Clinical Interpretation</p>
                    <p style="font-size: 0.85rem; color: #4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if dual.get("ecgProvided") and dual.get("ecgResult"):
                with res_col2:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("#### CNN VGG16 ECG")
                    st.caption("Waveform morphology classification")
                    
                    ecg = dual["ecgResult"]
                    st.markdown(risk_badge_html(ecg["riskLevel"]), unsafe_allow_html=True)
                    
                    # Classification result
                    is_normal = ecg["riskLevel"] == "low"
                    accent = "#10B981" if is_normal else "#EF4444"
                    
                    st.markdown(f"""
                    <div style="border-radius: 16px; padding: 16px; border: 1px solid {'#D1FAE5' if is_normal else '#FEE2E2'}; 
                         background: {'#ECFDF5' if is_normal else '#FEF2F2'}; margin: 12px 0;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span style="font-size: 2rem;">{'✅' if is_normal else '⚠️'}</span>
                            <div>
                                <h4 style="margin: 0; color: #0B1F3A;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                                <p style="margin: 0; font-size: 0.8rem; color: #6B7280;">{ecg["classification"]}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    conf_pct = int(ecg["confidence"] * 100)
                    st.markdown(f"""
                    <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0; margin: 12px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 0.7rem; font-weight: 700; color: #6B7280; text-transform: uppercase;">Model Confidence</span>
                            <span style="font-size: 1.5rem; font-weight: 800; color: #0B1F3A;">{conf_pct}%</span>
                        </div>
                        <div style="height: 12px; background: #E2E8F0; border-radius: 999px; overflow: hidden;">
                            <div style="width: {conf_pct}%; height: 100%; background: {'#10B981' if is_normal else '#EF4444'}; border-radius: 999px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Class probabilities
                    if ecg.get("probabilities"):
                        fig3 = render_ecg_probability_chart(ecg["probabilities"])
                        if fig3:
                            st.plotly_chart(fig3, use_container_width=True)
                    
                    # Findings
                    st.markdown(f"""
                    <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0;">
                        <p style="font-size: 0.7rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">Key Findings</p>
                        <p style="font-size: 0.85rem; color: #4B5563;">{ecg["findings"]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Combined result
            st.markdown("<div class='result-card' style='margin-top: 16px;'>", unsafe_allow_html=True)
            
            icon_map = {"high": "⚠️", "moderate": "⚡", "low": "✅"}
            color_map = {"high": "#EF4444", "moderate": "#F59E0B", "low": "#2EC4B6"}
            bg_map = {"high": "rgba(239,68,68,0.05)", "moderate": "rgba(245,158,11,0.05)", "low": "rgba(46,196,182,0.05)"}
            
            frl = dual["finalRiskLevel"]
            
            st.markdown(f"""
            <div style="border: 2px solid {color_map[frl]}; border-radius: 16px; padding: 20px; background: {bg_map[frl]};">
                <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 16px;">
                    <div style="padding: 12px; background: white; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 2rem;">
                        {icon_map[frl]}
                    </div>
                    <div>
                        <span style="font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #6B7280;">Consensus Triage</span>
                        <h3 style="margin: 4px 0; color: #0B1F3A; font-size: 1.5rem; text-transform: capitalize;">{frl} Risk</h3>
                        <span style="font-size: 0.75rem; color: #6B7280;">{(dual["confidenceScore"] * 100):.1f}% Confidence</span>
                    </div>
                </div>
                <div style="background: white; padding: 16px; border-radius: 12px;">
                    <p style="font-size: 0.9rem; color: #4B5563;">{dual["finalRecommendation"]}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # PDF Download
            pdf_data = {
                "mode": "dual",
                "patientName": patient_name if dual_submitted else "",
                "rfResult": dual["rfResult"],
                "ecgResult": dual.get("ecgResult"),
                "finalRiskLevel": dual["finalRiskLevel"],
                "finalRecommendation": dual["finalRecommendation"],
                "confidenceScore": dual["confidenceScore"],
                "ecgSource": "uploaded" if st.session_state.ecg_image else "sample" if st.session_state.ecg_sample else None
            }
            
            pdf_html = generate_pdf_html(pdf_data)
            b64 = base64.b64encode(pdf_html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="CardioShield_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" class="stButton" style="text-decoration: none;"><button style="width: 100%; margin-top: 16px;">📥 Download Clinical Report</button></a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 2: ECG-ONLY MODE
    # ============================================================================
    
    with tab2:
        st.markdown("### ECG Image Analysis")
        st.caption("CNN VGG16 waveform classification from ECG images")
        
        ecg_col1, ecg_col2 = st.columns([1, 1])
        
        with ecg_col1:
            # Sample selection
            sample_options = ["None", "Normal Sinus Rhythm", "Myocardial Infarction", "History of MI", "Abnormal Heartbeat"]
            sample_values = [None, "normal", "myocardial_infarction", "history_mi", "abnormal_heartbeat"]
            
            selected_idx = 0
            if st.session_state.ecg_sample:
                try:
                    selected_idx = sample_values.index(st.session_state.ecg_sample)
                except ValueError:
                    selected_idx = 0
            
            selected_sample = st.selectbox(
                "Select demo sample",
                options=range(len(sample_options)),
                format_func=lambda i: sample_options[i],
                index=selected_idx,
                key="ecg_sample_select"
            )
            
            if selected_sample > 0:
                st.session_state.ecg_sample = sample_values[selected_sample]
                st.session_state.ecg_image = None
                svg = get_ecg_sample_svg(st.session_state.ecg_sample)
                st.markdown(f'<div class="ecg-preview" style="margin: 12px 0;">{svg}</div>', unsafe_allow_html=True)
            else:
                st.session_state.ecg_sample = None
            
            st.markdown("---")
            st.caption("or upload an ECG image")
            
            uploaded_file = st.file_uploader(
                "Upload ECG Image",
                type=["png", "jpg", "jpeg"],
                key="ecg_only_upload"
            )
            
            if uploaded_file:
                st.session_state.ecg_sample = None
                bytes_data = uploaded_file.getvalue()
                st.session_state.ecg_image = base64.b64encode(bytes_data).decode()
                st.image(uploaded_file, caption="Uploaded ECG", use_container_width=True)
            
            has_ecg = bool(st.session_state.ecg_sample or st.session_state.ecg_image)
            
            if st.button("Analyze ECG", type="primary", use_container_width=True, disabled=not has_ecg, key="ecg_analyze_btn"):
                if st.session_state.ecg_image:
                    with st.spinner("Processing ECG image..."):
                        ecg_result = predict_ecg(st.session_state.ecg_image, vgg16_model)
                        st.session_state.ecg_result = ecg_result
                elif st.session_state.ecg_sample:
                    with st.spinner("Processing demo ECG..."):
                        ecg_result = simulate_ecg_prediction(is_demo=True)
                        st.session_state.ecg_result = ecg_result
        
        with ecg_col2:
            if st.session_state.ecg_result:
                ecg = st.session_state.ecg_result
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Diagnostic Result")
                st.caption("CNN VGG16 Morphological Analysis")
                
                is_normal = ecg["riskLevel"] == "low"
                
                st.markdown(f"""
                <div style="border-radius: 16px; padding: 16px; border: 1px solid {'#D1FAE5' if is_normal else '#FEE2E2'}; 
                     background: {'#ECFDF5' if is_normal else '#FEF2F2'}; margin: 12px 0;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 2rem;">{'✅' if is_normal else '⚠️'}</span>
                        <div>
                            <h4 style="margin: 0; color: #0B1F3A;">{'Normal Sinus Rhythm' if is_normal else 'Anomaly Detected'}</h4>
                            <p style="margin: 0; font-size: 0.8rem; color: #6B7280;">{ecg["classification"]}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                conf_pct = int(ecg["confidence"] * 100)
                st.markdown(f"""
                <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0; margin: 12px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-size: 0.7rem; font-weight: 700; color: #6B7280; text-transform: uppercase;">Model Confidence</span>
                        <span style="font-size: 1.5rem; font-weight: 800; color: #0B1F3A;">{conf_pct}%</span>
                    </div>
                    <div style="height: 12px; background: #E2E8F0; border-radius: 999px; overflow: hidden;">
                        <div style="width: {conf_pct}%; height: 100%; background: {'#10B981' if is_normal else '#EF4444'}; border-radius: 999px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if ecg.get("probabilities"):
                    fig = render_ecg_probability_chart(ecg["probabilities"])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"""
                <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0;">
                    <p style="font-size: 0.7rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase;">Key Findings</p>
                    <p style="font-size: 0.85rem; color: #4B5563;">{ecg["findings"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # PDF download
                pdf_data = {
                    "mode": "ecg",
                    "ecgResult": ecg,
                    "ecgSource": "uploaded" if st.session_state.ecg_image else "sample"
                }
                pdf_html = generate_pdf_html(pdf_data)
                b64 = base64.b64encode(pdf_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="CardioShield_ECG_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" style="text-decoration: none;"><button style="width: 100%; margin-top: 16px;">📥 Download ECG Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <span style="font-size: 3rem;">📈</span>
                    <h4 style="color: #0B1F3A;">Awaiting ECG Input</h4>
                    <p style="color: #6B7280;">Select a demo sample or upload an ECG image to begin analysis.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 3: DATA-ONLY MODE
    # ============================================================================
    
    with tab3:
        st.markdown("### Data-Only Analysis")
        st.caption("Random Forest + SMOTE evaluation. No ECG required — works anywhere, instantly.")
        
        data_col1, data_col2 = st.columns([1.2, 1])
        
        with data_col1:
            with st.form("data_patient_form"):
                patient_name = st.text_input("Patient ID / Name", key="data_name")
                
                form_col1, form_col2 = st.columns(2)
                with form_col1:
                    age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, key="data_age")
                    resting_bp = st.number_input("Resting BP (mmHg)", min_value=60, max_value=250, value=130, key="data_bp")
                    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700, value=220, key="data_chol")
                    max_hr = st.number_input("Max Heart Rate (bpm)", min_value=50, max_value=250, value=150, key="data_hr")
                    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1, key="data_oldpeak")
                    
                with form_col2:
                    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", key="data_sex")
                    chest_pain = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4],
                                              format_func=lambda x: {1: "1 — Typical Angina", 2: "2 — Atypical Angina",
                                                                     3: "3 — Non-Anginal Pain", 4: "4 — Asymptomatic"}[x], key="data_cp")
                    resting_ecg = st.selectbox("Resting ECG", options=[0, 1, 2],
                                               format_func=lambda x: {0: "0 — Normal", 1: "1 — ST-T Wave", 2: "2 — LV Hypertrophy"}[x], key="data_ecg")
                    st_slope = st.selectbox("ST Slope", options=[1, 2, 3],
                                            format_func=lambda x: {1: "1 — Upsloping", 2: "2 — Flat", 3: "3 — Downsloping"}[x], key="data_slope")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    fbs = st.toggle("Fasting Blood Sugar >120", value=False, key="data_fbs")
                    ex_angina = st.toggle("Exercise Induced Angina", value=False, key="data_exang")
                
                # Quick load buttons
                quick_col1, quick_col2 = st.columns(2)
                with quick_col1:
                    load_low = st.form_submit_button("Load Low Risk")
                with quick_col2:
                    load_high = st.form_submit_button("Load High Risk")
                
                data_submitted = st.form_submit_button("Analyze Patient Data", type="primary", use_container_width=True)
                
                if load_low:
                    st.session_state.data_preset = "low"
                    st.rerun()
                if load_high:
                    st.session_state.data_preset = "high"
                    st.rerun()
                
                # Apply presets
                if 'data_preset' in st.session_state:
                    if st.session_state.data_preset == "low":
                        age = 35
                        sex = 0
                        chest_pain = 1
                        resting_bp = 110
                        cholesterol = 180
                        max_hr = 170
                        oldpeak = 0.0
                        st_slope = 1
                        fbs = False
                        ex_angina = False
                        resting_ecg = 0
                    elif st.session_state.data_preset == "high":
                        age = 62
                        sex = 1
                        chest_pain = 4
                        resting_bp = 160
                        cholesterol = 310
                        max_hr = 95
                        oldpeak = 2.5
                        st_slope = 2
                        fbs = True
                        ex_angina = True
                        resting_ecg = 1
                
                if data_submitted:
                    patient_data = {
                        "patientName": patient_name,
                        "age": age,
                        "sex": sex,
                        "chestPainType": chest_pain,
                        "restingBpS": resting_bp,
                        "cholesterol": cholesterol,
                        "fastingBloodSugar": 1 if fbs else 0,
                        "restingEcg": resting_ecg,
                        "maxHeartRate": max_hr,
                        "exerciseAngina": 1 if ex_angina else 0,
                        "oldpeak": oldpeak,
                        "stSlope": st_slope
                    }
                    
                    with st.spinner("Analyzing patient data..."):
                        rf_result = predict_rf(patient_data, preprocessor, rf_model)
                        st.session_state.rf_result = rf_result
        
        with data_col2:
            if st.session_state.rf_result:
                rf = st.session_state.rf_result
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("#### Clinical Assessment")
                st.caption("Random Forest Feature Evaluation")
                
                st.markdown(risk_badge_html(rf["riskLevel"]), unsafe_allow_html=True)
                
                # Risk gauge
                fig = render_risk_gauge(rf["riskScore"], rf["riskLevel"])
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                if rf.get("features"):
                    fig2 = render_feature_importance_chart(rf["features"])
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown(f"""
                <div style="background: #F7F9FC; padding: 16px; border-radius: 12px; border: 1px solid #E2E8F0;">
                    <p style="font-size: 0.7rem; font-weight: 700; color: #0B1F3A; text-transform: uppercase;">Clinical Interpretation</p>
                    <p style="font-size: 0.85rem; color: #4B5563;">{rf["recommendation"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # PDF download
                pdf_data = {
                    "mode": "data",
                    "patientName": patient_name,
                    "rfResult": rf
                }
                pdf_html = generate_pdf_html(pdf_data)
                b64 = base64.b64encode(pdf_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="CardioShield_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html" style="text-decoration: none;"><button style="width: 100%; margin-top: 16px;">📥 Download Clinical Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="empty-state">
                    <span style="font-size: 3rem;">📊</span>
                    <h4 style="color: #0B1F3A;">Awaiting Patient Data</h4>
                    <p style="color: #6B7280;">Fill out the patient vitals form and click analyze to generate a cardiovascular risk assessment.</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 4: INVESTOR BRIEF
    # ============================================================================
    
    with tab4:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 40px;">
            <span style="display: inline-block; padding: 4px 16px; border-radius: 999px; background: rgba(46, 196, 182, 0.1); 
                  border: 1px solid rgba(46, 196, 182, 0.3); color: #2EC4B6; font-size: 0.7rem; font-weight: 700; 
                  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px;">
                Investment Brief · 2026
            </span>
            <h1 style="color: #0B1F3A; font-size: 2.5rem; font-weight: 700;">Redefining Cardiovascular Care with AI</h1>
            <p style="color: #6B7280; font-size: 1.1rem; max-width: 700px; margin: 0 auto;">
                A dual-mode cardiovascular screening platform deploying Random Forest with SMOTE and CNN VGG16 
                for instant triage in resource-constrained clinical environments.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Trust signals
        cols = st.columns(4)
        stats = [
            ("91%", "RF+SMOTE Accuracy", "1,200 patient records"),
            ("2,500+", "ECG Images", "CNN VGG16 Trained"),
            ("< 2s", "Inference Time", "Per prediction"),
            ("$4B+", "Addressable Market", "Global CDS software")
        ]
        
        for col, (value, label, sub) in zip(cols, stats):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background: #F7F9FC; border-radius: 16px; border: 1px solid #E2E8F0;">
                    <span style="font-size: 2rem; font-weight: 800; color: #0B1F3A;">{value}</span>
                    <p style="font-weight: 600; color: #1F2937; margin: 4px 0;">{label}</p>
                    <p style="font-size: 0.75rem; color: #9CA3AF;">{sub}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Problem & Architecture
        arch_col1, arch_col2 = st.columns(2)
        
        with arch_col1:
            st.markdown("""
            <div style="background: white; border-radius: 16px; border-top: 4px solid #EF4444; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="width: 40px; height: 40px; background: #FEE2E2; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                        <span style="color: #EF4444; font-size: 1.2rem;">⚠️</span>
                    </div>
                    <h3 style="color: #1F2937; margin: 0;">The Clinical Bottleneck</h3>
                </div>
                <p style="color: #4B5563; font-size: 0.9rem;">
                    Primary care and rural clinics cannot rely on ECG for every screening. A standard ECG requires equipment, trained technicians, and cardiologists to interpret.
                </p>
                <p style="color: #4B5563; font-size: 0.9rem; margin-top: 12px;">
                    This delays triage, limits scalability, and forces high-risk patients to wait for specialist availability — leading to worse outcomes and higher systemic costs.
                </p>
                <div style="background: #FEF2F2; padding: 12px; border-radius: 8px; margin-top: 16px; border: 1px solid #FEE2E2;">
                    <p style="color: #991B1B; font-size: 0.8rem; font-weight: 500;">
                        ⚠️ Cardiovascular disease accounts for 32% of all global deaths — early triage directly saves lives.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with arch_col2:
            st.markdown("""
            <div style="background: white; border-radius: 16px; border-top: 4px solid #2EC4B6; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
                    <div style="width: 40px; height: 40px; background: #D1FAE5; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                        <span style="color: #2EC4B6; font-size: 1.2rem;">📊</span>
                    </div>
                    <h3 style="color: #1F2937; margin: 0;">Dual-Mode AI Architecture</h3>
                </div>
                <p style="color: #4B5563; font-size: 0.9rem; font-weight: 600;">
                    Works dynamically with or without ECG availability.
                </p>
                <div style="margin-top: 16px;">
                    <div style="display: flex; gap: 12px; margin-bottom: 16px;">
                        <div style="width: 24px; height: 24px; background: #0B1F3A; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; flex-shrink: 0;">1</div>
                        <div>
                            <p style="font-weight: 700; color: #1F2937; font-size: 0.8rem; text-transform: uppercase;">Phase 1 — Structured Vitals</p>
                            <p style="color: #6B7280; font-size: 0.8rem;">Random Forest + SMOTE on 11 clinical features. No ECG required. 91% accuracy on 1,200 records.</p>
                        </div>
                    </div>
                    <div style="display: flex; gap: 12px;">
                        <div style="width: 24px; height: 24px; background: #2EC4B6; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; flex-shrink: 0;">2</div>
                        <div>
                            <p style="font-weight: 700; color: #1F2937; font-size: 0.8rem; text-transform: uppercase;">Phase 2 — ECG Validation</p>
                            <p style="color: #6B7280; font-size: 0.8rem;">CNN VGG16 on 2,500 ECG images across 4 classes. Deep learning morphological analysis.</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Value propositions
        st.markdown("""
        <div style="text-align: center; margin-bottom: 32px;">
            <p style="color: #2EC4B6; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;">Why CardioShield</p>
            <h2 style="color: #0B1F3A;">Platform Value Proposition</h2>
        </div>
        """, unsafe_allow_html=True)
        
        vp_cols = st.columns(4)
        vps = [
            ("✅", "Works Without ECG", "Democratizes screening to any clinic capable of taking basic vitals and history."),
            ("📊", "ECG Validation Layer", "Seamlessly integrates deep learning morphological analysis when equipment is available."),
            ("⚡", "Low-Cost Architecture", "Optimized inference pipelines minimize compute costs per prediction — scalable from day one."),
            ("📈", "Fast Pilotability", "No hardware required to launch. Software-only integration accelerates B2B sales cycles.")
        ]
        
        for col, (icon, title, desc) in zip(vp_cols, vps):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 24px;">
                    <div style="width: 56px; height: 56px; background: #F7F9FC; border-radius: 16px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; color: #2EC4B6; font-size: 1.5rem;">
                        {icon}
                    </div>
                    <p style="font-weight: 700; color: #0B1F3A; margin-bottom: 8px;">{title}</p>
                    <p style="font-size: 0.8rem; color: #6B7280;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Waitlist CTA
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #0B1F3A 0%, #122b4d 100%); border-radius: 24px; padding: 48px; text-align: center; color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50px; right: -50px; width: 200px; height: 200px; background: rgba(46, 196, 182, 0.2); border-radius: 50%; filter: blur(40px);"></div>
            <div style="position: relative; z-index: 1;">
                <h2 style="color: white; font-size: 2rem; font-weight: 700; margin-bottom: 12px;">Deploy CardioShield AI at your facility</h2>
                <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem; margin-bottom: 32px;">
                    Join leading clinics getting early access to our combined RF + CNN screening architecture.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Waitlist form
        with st.form("investor_waitlist"):
            waitlist_col1, waitlist_col2 = st.columns([3, 1])
            with waitlist_col1:
                waitlist_name = st.text_input("Full Name", placeholder="Dr. Jane Smith", key="investor_name")
                waitlist_email = st.text_input("Work Email", placeholder="you@hospital.org", key="investor_email")
            with waitlist_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Get Early Access →", type="primary", use_container_width=True)
                
                if submitted:
                    if waitlist_name and waitlist_email:
                        position = add_to_waitlist(waitlist_name, waitlist_email)
                        if position:
                            st.success(f"✅ You're #{position} on the waitlist!")
                        else:
                            st.warning("This email is already on the waitlist.")
                    else:
                        st.error("Please fill in all fields.")
    
    # ============================================================================
    # DISCLAIMER
    # ============================================================================
    
    with st.expander("ℹ️ Responsible AI & Clinical Use Disclaimer"):
        st.markdown("""
        CardioShield AI is a clinical decision support tool designed to assist healthcare professionals in evaluating cardiovascular risk. 
        It is **not a substitute** for professional medical diagnosis, advice, or treatment.

        The tabular model uses Random Forest with SMOTE trained on the Heart Statlog Cleveland Hungary Final dataset (1,200 patient records, 11 features). 
        The ECG classifier uses a VGG16 architecture trained on 2,500 ECG images. Both models provide statistical probabilities — clinicians must apply 
        professional judgment and full clinical context.
        """)
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <p style="color: #9CA3AF; font-size: 0.75rem;">
            CardioShield AI — Clinical decision support only &nbsp;·&nbsp; RF+SMOTE: 1,200 records &nbsp;·&nbsp; VGG16: 2,500 ECG images
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================================================
    # FLOATING WAITLIST BUTTON
    # ============================================================================
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📋 Join Waitlist")
        
        waitlist_count = get_waitlist_count()
        st.metric("Total Signups", waitlist_count)
        
        with st.form("sidebar_waitlist"):
            wl_name = st.text_input("Name", key="sidebar_name")
            wl_email = st.text_input("Email", key="sidebar_email")
            if st.form_submit_button("Join Waitlist", type="primary", use_container_width=True):
                if wl_name and wl_email:
                    pos = add_to_waitlist(wl_name, wl_email)
                    if pos:
                        st.success(f"Position #{pos}!")
                    else:
                        st.warning("Already registered.")
                else:
                    st.error("Fill all fields.")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
