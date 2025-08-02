import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import torch

def plot_predictions(actual_prices, predicted_prices, future_predictions=None, future_dates=None, 
                    validation_dates=None, ticker='Stock', save_path='stock_prediction.png'):
    """
    Create a comprehensive visualization of actual vs predicted prices with optional future predictions
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual prices
    plt.plot(actual_prices.index, actual_prices, label='Actual Prices', color='blue')
    
    # Plot validation dates and predicted prices
    if validation_dates is not None and len(predicted_prices) > 0:
        plt.plot(validation_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
    
    # Plot future predictions
    if future_predictions is not None and future_dates is not None:
        plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')
    
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_prediction_error(
    stock_data, 
    predictions, 
    confidence_intervals, 
    X_train, 
    X_test, 
    y_train, 
    y_test
):
    """
    Create a plot showing prediction errors and confidence intervals
    
    Args:
        stock_data (pd.DataFrame): Original stock data
        predictions (np.ndarray): Model predictions
        confidence_intervals (dict): Confidence interval data
        X_train (np.ndarray): Training input features
        X_test (np.ndarray): Test input features
        y_train (np.ndarray): Training target values
        y_test (np.ndarray): Test target values
    
    Returns:
        matplotlib.figure.Figure: Error visualization plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions, dtype=float)
    y_test = np.array(y_test, dtype=float)
    
    # Calculate prediction errors
    errors = np.abs(predictions.flatten() - y_test.flatten())
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Subplot for errors
    plt.subplot(2, 1, 1)
    plt.title('Prediction Errors', fontsize=15)
    plt.plot(errors, label='Absolute Error', color='red')
    plt.xlabel('Prediction Index')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.grid(True)
    
    # Subplot for error distribution
    plt.subplot(2, 1, 2)
    plt.title('Error Distribution', fontsize=15)
    plt.hist(errors, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Add error statistics
    plt.figtext(0.02, 0.02, 
        f"Mean Error: {np.mean(errors):.4f}\n" +
        f"Median Error: {np.median(errors):.4f}\n" +
        f"Max Error: {np.max(errors):.4f}",
        fontsize=10, 
        verticalalignment='bottom'
    )
    
    plt.tight_layout()
    return plt.gcf()

def plot_risk_assessment(risk_assessment):
    """
    Create a plot visualizing risk assessment metrics
    
    Args:
        risk_assessment (dict): Risk assessment metrics
    
    Returns:
        matplotlib.figure.Figure: Risk assessment visualization plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Bar plot for risk metrics
    metrics = list(risk_assessment.keys())
    values = list(risk_assessment.values())
    
    plt.bar(metrics, values, color=['blue', 'red', 'green', 'orange'])
    plt.title('Risk Assessment Metrics', fontsize=15)
    plt.xlabel('Risk Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def save_plots(plots):
    """
    Save multiple plots to the plots directory
    
    Args:
        plots (list): List of matplotlib figures to save
    """
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Save each plot
    for i, fig in enumerate(plots):
        fig.savefig(f'plots/plot_{i+1}_comprehensive.png', dpi=300)
        plt.close(fig)

def save_plots(plot, filename):
    """Save generated plots"""
    os.makedirs('plots', exist_ok=True)
    plot.savefig(f'plots/{filename}.png', dpi=300)
    plt.close(plot)
