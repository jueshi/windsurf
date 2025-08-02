import numpy as np
import torch
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.preprocessing import MinMaxScaler

def generate_predictions(model, X_test, forecast_horizon=30, scalers=None):
    """
    Generate predictions using the trained model
    
    Args:
        model (torch.nn.Module): Trained LSTM model
        X_test (np.ndarray): Input test data
        forecast_horizon (int): Number of future time steps to predict
        scalers (list, optional): List of scalers used for normalization
    
    Returns:
        tuple: Test predictions and future predictions
    """
    # Set model to evaluation mode
    model.eval()
    
    # Ensure input is torch tensor
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Disable gradient computation
    with torch.no_grad():
        # Predict for the entire test set (validation period)
        test_predictions = []
        
        # Slide through the entire test data
        for i in range(len(X_test)):
            # Get current sequence
            current_sequence = X_test[i]
            
            # Convert to tensor and predict
            current_sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            current_prediction = model(current_sequence_tensor).numpy()[0]
            
            # Store prediction
            test_predictions.append(current_prediction)
        
        # Convert to numpy array
        test_predictions = np.array(test_predictions)
        
        # Prepare for future predictions beyond validation period
        last_sequence = X_test[-1]  # Last sequence from test data
        future_predictions = []
        
        # Predict future time steps
        X_future = torch.FloatTensor(last_sequence).unsqueeze(0)
        future_predictions = model(X_future).detach().numpy().flatten()
        
        # If the number of future predictions doesn't match forecast_horizon, pad or truncate
        if len(future_predictions) < forecast_horizon:
            future_predictions = np.pad(future_predictions, 
                                        (0, forecast_horizon - len(future_predictions)), 
                                        mode='constant', 
                                        constant_values=future_predictions[-1])
        elif len(future_predictions) > forecast_horizon:
            future_predictions = future_predictions[:forecast_horizon]
    
    # Print diagnostic information
    print("\nPrediction Generation Diagnostics:")
    print("Input test data shape:", X_test.shape)
    print("Test predictions shape:", test_predictions.shape)
    print("Future predictions shape:", len(future_predictions))
    
    print("\nTest Predictions Range:")
    print("Min:", np.min(test_predictions))
    print("Max:", np.max(test_predictions))
    print("Mean:", np.mean(test_predictions))
    
    print("\nFuture Predictions Range:")
    print("Min:", np.min(future_predictions))
    print("Max:", np.max(future_predictions))
    print("Mean:", np.mean(future_predictions))
    
    return test_predictions, np.array(future_predictions)

def rescale_predictions(predictions, original_data, scaler):
    """
    Rescale predictions to original data range
    
    Args:
        predictions (np.ndarray): Normalized predictions
        original_data (np.ndarray): Original stock price data
        scaler (MinMaxScaler): Scaler used for normalization
    
    Returns:
        np.ndarray: Rescaled predictions
    """
    try:
        # Ensure predictions are 2D for inverse_transform
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Inverse transform using the scaler
        if scaler:
            rescaled_predictions = scaler.inverse_transform(predictions)
        else:
            # Fallback manual scaling if no scaler
            min_price = np.min(original_data[:, 0])
            max_price = np.max(original_data[:, 0])
            rescaled_predictions = predictions * (max_price - min_price) + min_price
        
        return rescaled_predictions.flatten()
    
    except Exception as e:
        print(f"Error during rescaling: {e}")
        # Fallback to original predictions if scaling fails
        return predictions.flatten()

def calculate_confidence_intervals(predictions, confidence_level=0.95):
    """Calculate confidence intervals for predictions"""
    # Ensure predictions is a numpy array
    predictions = np.array(predictions).flatten()
    
    # Calculate standard error
    std_error = np.std(predictions)
    
    # Calculate confidence interval
    lower_bound = predictions - (std_error * 1.96)  # 95% confidence interval
    upper_bound = predictions + (std_error * 1.96)
    
    return {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def assess_risk(predictions, stock_data):
    """
    Assess risk based on model predictions
    
    Args:
        predictions (np.ndarray): Model predictions
        stock_data (pd.DataFrame): Original stock data
    
    Returns:
        dict: Risk assessment metrics
    """
    # Convert predictions to float
    predictions = np.array(predictions, dtype=float)
    
    # Volatility calculation
    returns = stock_data['Close'].pct_change()
    
    # Ensure numeric values
    risk_metrics = {
        'volatility': float(returns.std()),
        'max_drawdown': float((returns.cummax() - returns).max()),
        'prediction_variance': float(np.var(predictions)),
        'prediction_range': float(np.max(predictions) - np.min(predictions))
    }
    
    return risk_metrics

def log_predictions(predictions, confidence_intervals, risk_assessment, ticker):
    """Log predictions to a JSON file"""
    # Ensure predictions are numpy arrays
    predictions = np.array(predictions).flatten()
    
    # Prepare log data
    log_data = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions.tolist(),
        'confidence_intervals': {
            'lower_bound': confidence_intervals.get('lower_bound', []).tolist(),
            'upper_bound': confidence_intervals.get('upper_bound', []).tolist()
        },
        'risk_assessment': {
            'volatility': risk_assessment.get('volatility', 0),
            'max_drawdown': risk_assessment.get('max_drawdown', 0),
            'prediction_variance': risk_assessment.get('prediction_variance', 0),
            'prediction_range': risk_assessment.get('prediction_range', 0)
        }
    }
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Generate unique filename
    log_filename = f'logs/{ticker}_prediction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # Write log data to JSON file
    with open(log_filename, 'w') as log_file:
        json.dump(log_data, log_file, indent=4)
    
    # Optional: Create a CSV log for easier data analysis
    csv_filename = f'logs/{ticker}_prediction_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    log_df = pd.DataFrame({
        'Prediction': predictions,
        'Lower Bound': confidence_intervals.get('lower_bound', []),
        'Upper Bound': confidence_intervals.get('upper_bound', [])
    })
    log_df.to_csv(csv_filename, index=False)
    
    return log_filename
