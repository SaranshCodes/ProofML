# verifier.py
import numpy as np

def verify_certificate_data(X_sample, W, b, model_prediction, epsilon=1e-6):
    """
    Independently verifies the model prediction given the input X, weights W, and bias b.
    Returns a dictionary with verification results.
    """
    # Ensure numpy array
    if hasattr(X_sample, "values"):
        X_array = X_sample.values
    else:
        X_array = np.array(X_sample)
        
    # Independent recalculation
    p_verified = np.dot(W, X_array) + b
    
    # Check absolute difference against tolerance
    difference = abs(model_prediction - p_verified)
    is_valid = difference < epsilon
    
    return {
        "model_prediction": model_prediction,
        "verified_prediction": p_verified,
        "difference": difference,
        "is_valid": is_valid,
        "status": "VALID" if is_valid else "INVALID"
    }

if __name__ == "__main__":
    from data import load_data
    import joblib
    
    _, X_test, _, _ = load_data()
    model = joblib.load("models/model.pkl")
    
    sample = X_test.iloc[0]
    pred = model.predict(sample.values.reshape(1, -1))[0]
    
    res = verify_certificate_data(sample, model.coef_, model.intercept_, pred)
    
    print("=" * 50)
    print("INDEPENDENT VERIFIER TEST")
    print("=" * 50)
    print(f"Model Prediction    : {res['model_prediction']:.6f}")
    print(f"Verified Prediction : {res['verified_prediction']:.6f}")
    print(f"Difference          : {res['difference']:.2e}")
    print(f"Certificate Status  : {res['status']}")
    print("=" * 50)
# verifier.py (Extension for Corrupted Test)
# You can append or run this test function to demonstrate failure detection.

def test_corrupted_certificate(X_sample, W, b, model_prediction):
    """
    Demonstrates that modifying a coefficient causes the verifier 
    to reject the certificate.
    """
    print("\n" + "=" * 50)
    print("CORRUPTED CERTIFICATE EXPERIMENT")
    print("=" * 50)
    
    # 1. Test with correct parameters first
    valid_res = verify_certificate_data(X_sample, W, b, model_prediction)
    print(f"Original (Correct) Parameters:")
    print(f"  W = {W}, b = {b:.4f}")
    print(f"  Verification Result: {valid_res['status']} ✓")
    
    # 2. Tamper with the first coefficient (e.g., add 2.5)
    corrupted_W = W.copy()
    corrupted_W[0] += 2.5
    
    corrupted_res = verify_certificate_data(X_sample, corrupted_W, b, model_prediction)
    print(f"\nTampered (Corrupted) Parameters:")
    print(f"  W_corrupted = {corrupted_W}, b = {b:.4f}")
    print(f"  Absolute Difference : {corrupted_res['difference']:.4f}")
    print(f"  Verification Result : {corrupted_res['status']} ✗")
    print("=" * 50)

if __name__ == "__main__":
    from data import load_data
    import joblib
    
    _, X_test, _, _ = load_data()
    model = joblib.load("models/model.pkl")
    
    sample = X_test.iloc[0]
    pred = model.predict(sample.values.reshape(1, -1))[0]
    
    # Run corruption test
    test_corrupted_certificate(sample, model.coef_, model.intercept_, pred)