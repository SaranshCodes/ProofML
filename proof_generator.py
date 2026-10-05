# proof_generator.py
import os

def generate_certificate(prediction_result):
    """
    Takes the dictionary from predictor.py and formats it into a 
    human-readable mathematical certificate string.
    """
    fn = prediction_result["feature_names"]
    fv = prediction_result["feature_values"]
    W = prediction_result["coefficients"]
    b = prediction_result["bias"]
    
    model_pred = prediction_result["model_prediction"]
    manual_pred = prediction_result["manual_prediction"]
    diff = prediction_result["difference"]
    is_valid = prediction_result["is_consistent"]
    
    # Build dynamic strings for transformation steps
    terms_formula = " + ".join([f"w{i+1}*x{i+1}" for i in range(len(fn))]) + " + b"
    terms_calc = " + ".join([f"({W[i]:.4f} × {fv[i]:.4f})" for i in range(len(fn))]) + f" + {b:.4f}"
    
    # Individual multiplication products for intermediate expansion
    products = [W[i] * fv[i] for i in range(len(fn))]
    terms_sum = " + ".join([f"{p:.4f}" for p in products]) + f" + {b:.4f}"
    
    status_str = "VERIFIED" if is_valid else "FAILED"
    verification_symbol = "PASSED ✓" if is_valid else "FAILED ✗"
    
    certificate_text = f"""
=========================================
PROOF-AWARE ML CERTIFICATE
=========================================
Model: Linear Regression

Input Features:
{chr(10).join([f"  {fn[i]} = {fv[i]:.4f}" for i in range(len(fn))])}

Model Parameters:
{chr(10).join([f"  w{i+1} ({fn[i]}) = {W[i]:.4f}" for i in range(len(fn))])}
  b (intercept) = {b:.4f}

Mathematical Transformation:
  ŷ = {terms_formula}
  ŷ = {terms_calc}
  ŷ = {terms_sum}
  ŷ = {manual_pred:.4f}

ML Model Prediction:
  {model_pred:.6f}

Independent Calculation:
  {manual_pred:.6f}

Difference:
  {diff:.2e}

Verification:
  {verification_symbol}

Certificate Status:
  {status_str}
=========================================
"""
    return certificate_text

def save_certificate(certificate_text, output_path="certificates/certificate.txt"):
    """
    Saves the generated certificate string to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(certificate_text)
    print(f"Certificate saved successfully to {output_path}")

if __name__ == "__main__":
    from data import load_data
    from predictor import predict_with_manual_check
    
    _, X_test, _, _ = load_data()
    sample = X_test.iloc[0]
    res = predict_with_manual_check(sample)
    
    cert = generate_certificate(res)
    print(cert)
    save_certificate(cert)