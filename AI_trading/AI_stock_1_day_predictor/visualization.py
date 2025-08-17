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
    
    Args:
        actual_prices (np.array): Actual stock prices during validation period
        predicted_prices (np.array): Predicted stock prices for validation period
        future_predictions (np.array, optional): Predicted future stock prices
        future_dates (pd.DatetimeIndex, optional): Dates for future predictions
        validation_dates (pd.DatetimeIndex, optional): Dates for validation period
        ticker (str, optional): Stock ticker symbol
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(16, 8))
    
    # Ensure inputs are numpy arrays
    actual_prices = np.array(actual_prices).flatten()
    predicted_prices = np.array(predicted_prices).flatten()
    
    # Validation period plot
    plt.subplot(2, 1, 1)
    plt.title(f'{ticker} Stock Price: Actual vs Predicted (Validation Period)', fontsize=15)
    plt.plot(validation_dates, actual_prices, label='Actual Prices', color='blue')
    plt.plot(validation_dates, predicted_prices, label='Predicted Prices', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Future predictions plot
    plt.subplot(2, 1, 2)
    plt.title(f'{ticker} Stock Price: Future Predictions', fontsize=15)
    
    # Combine validation and future dates
    if future_predictions is not None and future_dates is not None:
        # Extend actual prices to include last validation price for continuity
        extended_actual = np.concatenate([actual_prices, [actual_prices[-1]]])
        extended_dates = np.concatenate([validation_dates, [validation_dates[-1]], future_dates])
        
        # Ensure dates and prices are the same length
        if len(extended_dates) > len(extended_actual):
            extended_dates = extended_dates[:len(extended_actual)]
        elif len(extended_actual) > len(extended_dates):
            extended_actual = extended_actual[:len(extended_dates)]
        
        # Plot extended actual prices and future predictions
        plt.plot(extended_dates, extended_actual, label='Actual Prices', color='blue')
        plt.plot(future_dates, future_predictions, label='Future Predictions', color='green', linestyle='--')
        
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        plt.savefig(f'plots/{save_path}')
    plt.show()

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
