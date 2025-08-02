"""Utility functions for the MLOps pipeline"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Load and prepare California Housing dataset
    
    Args:
        test_size (float): Proportion of dataset for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def calculate_loss(y_true, y_pred):
    """
    Calculate Mean Squared Error loss
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        float: MSE loss
    """
    return np.mean((y_true - y_pred) ** 2)


def quantize_weights(weights, bits=8):
    """
    Quantize weights to specified bit precision
    
    Args:
        weights: Array of weights to quantize
        bits: Number of bits for quantization
        
    Returns:
        tuple: (quantized_weights, scale, zero_point)
    """
    # Calculate scale and zero point
    w_min, w_max = weights.min(), weights.max()
    qmin, qmax = 0, (2 ** bits) - 1
    
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = qmin - w_min / scale
    zero_point = np.clip(np.round(zero_point), qmin, qmax).astype(np.uint8)
    
    # Quantize
    quantized = np.clip(np.round(weights / scale + zero_point), qmin, qmax).astype(np.uint8)
    
    return quantized, scale, zero_point


def dequantize_weights(quantized_weights, scale, zero_point):
    """
    Dequantize weights back to float32
    
    Args:
        quantized_weights: Quantized weights
        scale: Scale factor
        zero_point: Zero point
        
    Returns:
        numpy.ndarray: Dequantized weights
    """
    return scale * (quantized_weights.astype(np.float32) - zero_point)