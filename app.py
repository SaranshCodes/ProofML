# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from predictor import predict_with_manual_check
from proof_generator import generate_certificate
from verifier import verify_certificate_data

# Page configuration
st.set_page_config(
    page_title="ProofML – Proof-Aware Machine Learning",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_cached_model():
    try:
        model = joblib.load("models/model.pkl")
        return model
    except FileNotFoundError:
        return None

model = load_cached_model()

st.title("ProofML – Proof-Aware Machine Learning")
st.markdown("A lightweight demonstration framework for generating mathematical certificates of ML model predictions.")

if model is None:
    st.error("Model not found! Please run `train_model.py` first to generate `models/model.pkl`.")
else:
    st.sidebar.header("Input Features")
    st.sidebar.markdown("Adjust feature values for `bmi` and `bp`:")
    
    # Default values based on dataset means/ranges
    bmi_input = st.sidebar.number_input("Feature 1: BMI", value=0.05, step=0.01, format="%.4f")
    bp_input = st.sidebar.number_input("Feature 2: Blood Pressure (bp)", value=0.02, step=0.01, format="%.4f")
    
    # Optional tampering toggle for viva demonstration
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced / Viva Test")
    tamper_checkbox = st.sidebar.checkbox("Corrupt Coefficient (Test Failure)")
    
    if st.button("Generate Prediction & Certificate"):
        # Package input
        sample = pd.Series([bmi_input, bp_input], index=['bmi', 'bp'])
        
        # Run prediction & manual check
        res = predict_with_manual_check(sample)
        
        if tamper_checkbox:
            # Tamper with coefficients for demonstration
            res["coefficients"][0] += 2.5
            res["manual_prediction"] = np.dot(res["coefficients"], sample.values) + res["bias"]
            res["difference"] = abs(res["model_prediction"] - res["manual_prediction"])
            res["is_consistent"] = res["difference"] < 1e-6

        # Generate certificate string
        cert_text = generate_certificate(res)
        
        # Display Results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="ML Model Prediction", value=f"{res['model_prediction']:.4f}")
        with col2:
            status_color = "normal" if res["is_consistent"] else "inverse"
            status_label = "VERIFIED ✓" if res["is_consistent"] else "FAILED ✗"
            st.metric(label="Certificate Status", value=status_label, delta=f"Diff: {res['difference']:.2e}")
            
        st.markdown("### Mathematical Derivation & Certificate")
        st.code(cert_text, language="text")
        
        if res["is_consistent"]:
            st.success("✓ Prediction successfully verified mathematically!")
        else:
            st.error("✗ Verification Failed! Mathematical inconsistency detected between model prediction and certificate parameters.")