
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import load_and_prepare_data, calculate_loss


def train_model():
    print("Loading and preparing data ")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    print("Training Linear Regression model ")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_loss = calculate_loss(y_train, y_train_pred)
    test_loss = calculate_loss(y_test, y_test_pred)
    
    print(f"Training R² Score: {train_r2:.4f}")
    print(f"Testing R² Score: {test_r2:.4f}")
    print(f"Training Loss (MSE): {train_loss:.4f}")
    print(f"Testing Loss (MSE): {test_loss:.4f}")
    
    # Save model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'linear_regression_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Save test data for later use
    joblib.dump((X_test, y_test), 'test_data.joblib')
    
    print("Model training completed successfully!")
    return model, scaler, test_r2


if __name__ == "__main__":
    train_model()