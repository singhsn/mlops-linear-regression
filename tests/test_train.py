"""Unit tests for training pipeline"""
import pytest
import numpy as np
import joblib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from utils import load_and_prepare_data
from train import train_model


class TestTraining:
    """Test class for training functionality"""
    
    def test_dataset_loading(self):
        """Test if dataset loads correctly"""
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        
        # Check if data is loaded
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert scaler is not None
        
        # Check data shapes
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check if features are scaled (mean should be close to 0)
        assert abs(np.mean(X_train)) < 0.1
        
        print("✓ Dataset loading test passed")
    
    def test_model_creation(self):
        """Test if model is LinearRegression instance"""
        model = LinearRegression()
        assert isinstance(model, LinearRegression)
        print("✓ Model creation test passed")
    
    def test_model_training(self):
        """Test if model trains correctly"""
        # Train the model
        model, scaler, test_r2 = train_model()
        
        # Check if model is trained (coefficients exist)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None
        
        # Check if coefficients have expected shape
        assert len(model.coef_) == 8  # California housing has 8 features
        
        print("✓ Model training test passed")
    
    def test_r2_score_threshold(self):
        """Test if R² score exceeds minimum threshold"""
        # Load the trained model and test it
        if os.path.exists('linear_regression_model.joblib'):
            model = joblib.load('linear_regression_model.joblib')
            scaler = joblib.load('scaler.joblib')
            X_test, y_test = joblib.load('test_data.joblib')
            
            predictions = model.predict(X_test)
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, predictions)
            
            # R² score should be at least 0.5 for a reasonable model
            MIN_R2_THRESHOLD = 0.5
            assert r2 >= MIN_R2_THRESHOLD, f"R² score {r2:.4f} is below minimum threshold {MIN_R2_THRESHOLD}"
            
            print(f"✓ R² score threshold test passed (R² = {r2:.4f})")
        else:
            # If model doesn't exist, train it first
            model, scaler, test_r2 = train_model()
            MIN_R2_THRESHOLD = 0.5
            assert test_r2 >= MIN_R2_THRESHOLD, f"R² score {test_r2:.4f} is below minimum threshold {MIN_R2_THRESHOLD}"
            print(f"✓ R² score threshold test passed (R² = {test_r2:.4f})")
    
    def test_model_file_creation(self):
        """Test if model files are created"""
        # Ensure model is trained
        if not os.path.exists('linear_regression_model.joblib'):
            train_model()
        
        # Check if all required files exist
        assert os.path.exists('linear_regression_model.joblib')
        assert os.path.exists('scaler.joblib')
        assert os.path.exists('test_data.joblib')
        
        print("✓ Model file creation test passed")


if __name__ == "__main__":
    # Run tests manually
    test_instance = TestTraining()
    test_instance.test_dataset_loading()
    test_instance.test_model_creation()
    test_instance.test_model_training()
    test_instance.test_r2_score_threshold()
    test_instance.test_model_file_creation()
    print("All tests passed successfully!")