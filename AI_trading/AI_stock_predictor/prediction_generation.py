import numpy as np
import torch
import pandas as pd
from datetime import datetime
import json
import os

def generate_predictions(model, X_test, scaler, forecast_horizon):
    """Generate predictions using PyTorch model"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor)
        
        # Ensure predictions are numpy array
        predictions_np = predictions.numpy()
        
        # Reshape predictions to match scaler
        predictions_reshaped = predictions_np.reshape(-1, 1)
        
        # Inverse transform predictions
        predictions_rescaled = scaler.inverse_transform(predictions_reshaped)
    
    return predictions_rescaled

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
    """Assess risk based on predictions and historical data"""
    # Ensure predictions is a numpy array
    predictions = np.array(predictions).flatten()
    
    # Calculate volatility (standard deviation of returns)
    historical_returns = stock_data['Close'].pct_change()
    volatility = historical_returns.std()
    
    # Calculate prediction volatility
    prediction_volatility = np.std(predictions)
    
    # Simple risk scoring mechanism
    risk_score = (volatility + prediction_volatility) / 2
    
    # Categorize risk level
    if risk_score < 0.05:
        risk_level = 'Low'
    elif risk_score < 0.1:
        risk_level = 'Medium'
    else:
        risk_level = 'High'
    
    return {
        'volatility': float(volatility),
        'max_drawdown': float(historical_returns.min()),
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'prediction_volatility': float(prediction_volatility)
    }

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
            'risk_score': risk_assessment.get('risk_score', 0),
            'risk_level': risk_assessment.get('risk_level', 'Unknown'),
            'prediction_volatility': risk_assessment.get('prediction_volatility', 0)
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
