# data.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

def load_data(test_size=0.2, random_state=42):
    """
    Loads a real regression dataset from scikit-learn, selects 2 features,
    and splits it into training and testing sets.
    """
    # Load the standard diabetes dataset from scikit-learn
    diabetes = load_diabetes()
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    
    # For a clean, easily demonstrable mathematical certificate, 
    # we select just 2 features: 'bmi' (Body Mass Index) and 'bp' (Blood Pressure)
    # Target (y) is a quantitative measure of disease progression one year post-baseline
    X = df[['bmi', 'bp']]
    y = pd.Series(diabetes.target, name="target")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Dataset loaded successfully from Scikit-Learn!")
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
    print("\nSample X_test features:")
    print(X_test.head(3))