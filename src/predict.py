
import joblib
import numpy as np
from sklearn.metrics import r2_score
from utils import calculate_loss


def make_predictions():

    print("Loading model and test data")
    try:
        model = joblib.load('linear_regression_model.joblib')
        scaler = joblib.load('scaler.joblib')
        X_test, y_test = joblib.load('test_data.joblib')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return False
    
    print("Making predictions")
    predictions = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    loss = calculate_loss(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {loss:.4f}")
    
    # Show some sample predictions
    print(f"\nSample Predictions (first 10):")
    print("Actual\t\tPredicted\tDifference")
    print("-" * 40)
    for i in range(min(10, len(y_test))):
        diff = abs(y_test[i] - predictions[i])
        print(f"{y_test[i]:.4f}\t\t{predictions[i]:.4f}\t\t{diff:.4f}")
    
    print("Predictions completed successfully!")
    return True


if __name__ == "__main__":
    success = make_predictions()
    if not success:
        exit(1)