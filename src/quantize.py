
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import quantize_weights, dequantize_weights, calculate_loss


def quantize_model():
    
    print("Loading trained model...")
    try:
        model = joblib.load('linear_regression_model.joblib')
        scaler = joblib.load('scaler.joblib')
        X_test, y_test = joblib.load('test_data.joblib')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run train.py first to generate the model files.")
        return
    
    print("Extracting model parameters...")
    coef = model.coef_
    intercept = model.intercept_
    
    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    joblib.dump(raw_params, 'unquant_params.joblib')
    print("Raw parameters saved to unquant_params.joblib")
    
    # Quantize coefficients
    print("Quantizing coefficients...")
    quant_coef, coef_scale, coef_zero_point = quantize_weights(coef)
    
    # Quantize intercept (treating as single value array)
    intercept_array = np.array([intercept])
    quant_intercept, intercept_scale, intercept_zero_point = quantize_weights(intercept_array)
    
    # Save quantized parameters
    quantized_params = {
        'quant_coef': quant_coef,
        'coef_scale': coef_scale,
        'coef_zero_point': coef_zero_point,
        'quant_intercept': quant_intercept,
        'intercept_scale': intercept_scale,
        'intercept_zero_point': intercept_zero_point
    }
    joblib.dump(quantized_params, 'quant_params.joblib')
    print("Quantized parameters saved to quant_params.joblib")
    
    # Test quantized model
    print("Testing quantized model...")
    
    # Dequantize parameters
    dequant_coef = dequantize_weights(quant_coef, coef_scale, coef_zero_point)
    dequant_intercept = dequantize_weights(quant_intercept, intercept_scale, intercept_zero_point)[0]
    
    # Create new model with dequantized weights
    quantized_model = LinearRegression()
    quantized_model.coef_ = dequant_coef
    quantized_model.intercept_ = dequant_intercept
    
    # Make predictions with both models
    original_pred = model.predict(X_test)
    quantized_pred = quantized_model.predict(X_test)
    
    # Calculate metrics
    original_r2 = r2_score(y_test, original_pred)
    quantized_r2 = r2_score(y_test, quantized_pred)
    original_loss = calculate_loss(y_test, original_pred)
    quantized_loss = calculate_loss(y_test, quantized_pred)
    
    print(f"Original Model - R² Score: {original_r2:.4f}, Loss: {original_loss:.4f}")
    print(f"Quantized Model - R² Score: {quantized_r2:.4f}, Loss: {quantized_loss:.4f}")
    print(f"R² Score Difference: {abs(original_r2 - quantized_r2):.6f}")
    
    # Calculate compression ratio
    original_size = coef.nbytes + np.array([intercept]).nbytes
    quantized_size = quant_coef.nbytes + quant_intercept.nbytes
    compression_ratio = original_size / quantized_size
    
    print(f"Model size reduction: {original_size} bytes -> {quantized_size} bytes")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    print("Model quantization completed successfully!")


if __name__ == "__main__":
    quantize_model()