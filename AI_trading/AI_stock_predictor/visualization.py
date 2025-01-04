import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_predictions(historical_data, predictions, confidence_intervals):
    """Plot historical data and predictions with confidence intervals"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data
    ax.plot(historical_data.index, historical_data['Close'], 
             label='Historical Prices', color='blue')
    
    # Ensure predictions and confidence intervals are numpy arrays and flattened
    predictions = np.array(predictions).flatten()
    
    # Create future dates for predictions
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(predictions))
    
    # Plot predictions
    ax.scatter(future_dates, predictions, color='red', label='Prediction', marker='x')
    
    # Handle confidence intervals
    try:
        # Try to get lower and upper bounds, default to standard deviation if not available
        lower_bound = np.array(confidence_intervals.get('lower', predictions - np.std(predictions))).flatten()
        upper_bound = np.array(confidence_intervals.get('upper', predictions + np.std(predictions))).flatten()
        
        # Ensure bounds are the same length as predictions
        if len(lower_bound) != len(predictions):
            lower_bound = np.full_like(predictions, lower_bound[0])
        if len(upper_bound) != len(predictions):
            upper_bound = np.full_like(predictions, upper_bound[0])
        
        # Plot confidence intervals
        ax.fill_between(future_dates, lower_bound, upper_bound, 
                        color='orange', alpha=0.3, label='Confidence Interval')
    except Exception as e:
        print(f"Warning: Could not plot confidence intervals: {e}")
    
    ax.set_title('Stock Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/prediction_plot.png')
    plt.close(fig)
    
    return fig

def plot_risk_assessment(risk_metrics):
    """Visualize risk assessment metrics"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Safely extract metrics with defaults
    metrics = ['Volatility', 'Max Drawdown', 'Risk Score']
    values = [
        risk_metrics.get('volatility', 0), 
        risk_metrics.get('max_drawdown', 0), 
        risk_metrics.get('risk_score', 0)
    ]
    
    ax.bar(metrics, values, color=['red', 'orange', 'purple'])
    ax.set_title('Risk Assessment Metrics')
    ax.set_ylabel('Value')
    ax.grid(True, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    fig.savefig('plots/risk_assessment_plot.png')
    plt.close(fig)
    
    return fig

def save_plots(plot, filename):
    """Save generated plots"""
    os.makedirs('plots', exist_ok=True)
    plot.savefig(f'plots/{filename}.png')
    plt.close(plot)
