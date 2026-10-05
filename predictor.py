# predictor.py
import os
import joblib
import numpy as np
import pandas as pd

def predict_with_manual_check(X_sample, model_path="models/model.pkl"):
    """
    Loads the trained model, makes a prediction using scikit-learn, 
    and independently recalculates it using W^T * X + b.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
        
    # 1. Load trained model
    model = joblib.load(model_path)
    
    # Extract parameters
    W = model.coef_
    b = model.intercept_
    
    # Format input sample as a numpy array
    if isinstance(X_sample, pd.Series):
        X_array = X_sample.values
        feature_names = X_sample.index.tolist()
    elif isinstance(X_sample, pd.DataFrame):
        X_array = X_sample.iloc[0].values
        feature_names = X_sample.columns.tolist()
    else:
        X_array = np.array(X_sample)
        feature_names = [f"x{i+1}" for i in range(len(X_array))]
        
    # 2. Scikit-learn prediction (expects a 2D array)
    X_2d = X_array.reshape(1, -1)
    model_prediction = model.predict(X_2d)[0]
    
    # 3. Independent mathematical calculation: W^T * X + b
    manual_prediction = np.dot(W, X_array) + b
    
    # 4. Check consistency using tolerance epsilon = 10^-6
    epsilon = 1e-6
    difference = abs(model_prediction - manual_prediction)
    is_consistent = difference < epsilon
    
    result = {
        "feature_names": feature_names,
        "feature_values": X_array,
        "coefficients": W,
        "bias": b,
        "model_prediction": model_prediction,
        "manual_prediction": manual_prediction,
        "difference": difference,
        "is_consistent": is_consistent
    }
    
    return result

if __name__ == "__main__":
    from data import load_data
    _, X_test, _, _ = load_data()
    
    # Test with the first sample from our test set
    sample = X_test.iloc[0]
    res = predict_with_manual_check(sample)
    
    print("=" * 50)
    print("PREDICTOR & MANUAL CHECK TEST")
    print("=" * 50)
    print("Input Features:")
    for name, val in zip(res["feature_names"], res["feature_values"]):
        print(f"  {name} = {val:.4f}")
    print(f"\nModel Prediction    : {res['model_prediction']:.6f}")
    print(f"Manual Calculation  : {res['manual_prediction']:.6f}")
    print(f"Absolute Difference : {res['difference']:.2e}")
    print(f"Mathematical Match  : {res['is_consistent']}")
    print("=" * 50)