
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(test_size=0.2, random_state=42):
   
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


#Calculate Mean Squared Error loss
def calculate_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


#Quantize weights to specified bit precision
def quantize_weights(weights, bits=8):

    # Calculate scale and zero point
    w_min, w_max = weights.min(), weights.max()
    qmin, qmax = 0, (2 ** bits) - 1
    
    scale = (w_max - w_min) / (qmax - qmin)
    zero_point = qmin - w_min / scale
    zero_point = np.clip(np.round(zero_point), qmin, qmax).astype(np.uint8)
    
    # Quantize
    quantized = np.clip(np.round(weights / scale + zero_point), qmin, qmax).astype(np.uint8)
    
    return quantized, scale, zero_point

# Dequantize weights back to float32
def dequantize_weights(quantized_weights, scale, zero_point):
    return scale * (quantized_weights.astype(np.float32) - zero_point)