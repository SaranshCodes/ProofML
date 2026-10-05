# train_model.py
import os
import joblib
from sklearn.linear_model import LinearRegression
from data import load_data

def train_and_save_model():
    """
    Trains a Linear Regression model on the diabetes dataset features (bmi, bp),
    extracts coefficients and intercept, and saves the model to models/model.pkl.
    """
    # 1. Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. Initialize and train Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. Extract parameters
    coefficients = model.coef_
    intercept = model.intercept_
    feature_names = X_train.columns.tolist()
    
    print("=" * 50)
    print("MODEL TRAINING COMPLETED")
    print("=" * 50)
    print(f"Learned Intercept (b): {intercept:.4f}")
    print("Learned Coefficients (W):")
    for name, coef in zip(feature_names, coefficients):
        print(f"  {name}: {coef:.4f}")
        
    print("\nLearned Mathematical Equation:")
    eq_str = f"ŷ = {coefficients[0]:.4f} * {feature_names[0]} + {coefficients[1]:.4f} * {feature_names[1]} + {intercept:.4f}"
    print(f"  {eq_str}")
    print("=" * 50)
    
    # 4. Ensure models directory exists and save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}\n")
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_and_save_model()