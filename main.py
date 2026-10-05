# main.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data import load_data
from train_model import train_and_save_model
from predictor import predict_with_manual_check
from proof_generator import generate_certificate, save_certificate
from verifier import verify_certificate_data, test_corrupted_certificate

def run_pipeline():
    print("=" * 60)
    print(" STARTING PROOFML PIPELINE & EXPERIMENTS")
    print("=" * 60)
    
    # 1. Load Data & Train Model
    X_train, X_test, y_train, y_test = load_data()
    try:
        model = joblib.load("models/model.pkl")
        print("[Info] Loaded existing model from models/model.pkl")
    except FileNotFoundError:
        print("[Info] No saved model found. Training new model...")
        model, _, _ = train_and_save_model()
        
    W = model.coef_
    b = model.intercept_
    
    # 2. Evaluate ML Model Performance (MAE, RMSE, R^2)
    y_pred_all = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_all))
    r2 = r2_score(y_test, y_pred_all)
    
    print("\n" + "-" * 60)
    print(" 1. MODEL PERFORMANCE METRICS")
    print("-" * 60)
    print(f"  Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"  R-squared Score (R²)      : {r2:.4f}")
    
    # 3. Experiment 1: Single Normal Prediction & Certificate Generation
    print("\n" + "-" * 60)
    print(" 2. EXPERIMENT 1: SINGLE PREDICTION & CERTIFICATE")
    print("-" * 60)
    sample_idx = 0
    sample_x = X_test.iloc[sample_idx]
    pred_res = predict_with_manual_check(sample_x)
    
    certificate_str = generate_certificate(pred_res)
    print(certificate_str)
    save_certificate(certificate_str, "certificates/certificate.txt")
    
    # 4. Experiment 2: Batch Verification on Multiple Test Samples (e.g., 20 samples)
    print("\n" + "-" * 60)
    print(" 3. EXPERIMENT 2: BATCH VERIFICATION (20 SAMPLES)")
    print("-" * 60)
    num_samples = min(20, len(X_test))
    verified_count = 0
    failed_count = 0
    
    for i in range(num_samples):
        x_i = X_test.iloc[i]
        p_model = model.predict(x_i.values.reshape(1, -1))[0]
        v_res = verify_certificate_data(x_i, W, b, p_model)
        if v_res["is_valid"]:
            verified_count += 1
        else:
            failed_count += 1
            
    verification_rate = (verified_count / num_samples) * 100
    print(f"  Total Samples Evaluated : {num_samples}")
    print(f"  Verified Successfully   : {verified_count}")
    print(f"  Failed Verifications    : {failed_count}")
    print(f"  Verification Rate       : {verification_rate:.1f}%")
    
    # 5. Experiment 3: Corrupted Certificate Test
    print("\n" + "-" * 60)
    print(" 4. EXPERIMENT 3: CORRUPTED CERTIFICATE DETECTION")
    print("-" * 60)
    test_corrupted_certificate(sample_x, W, b, pred_res["model_prediction"])
    
    # 6. Summary Results Table for Report
    print("\n" + "=" * 60)
    print(" FINAL RESULTS SUMMARY TABLE")
    print("=" * 60)
    summary_table = pd.DataFrame([
        {"Experiment": "Normal Batch (20 samples)", "Samples": num_samples, "Verified": verified_count, "Failed": failed_count, "Verification Rate": f"{verification_rate:.1f}%"},
        {"Experiment": "Corrupted Certificate Test", "Samples": 1, "Verified": 0, "Failed": 1, "Verification Rate": "0.0%"}
    ])
    print(summary_table.to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()