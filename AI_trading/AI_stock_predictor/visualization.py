import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def plot_predictions(
    stock_data, 
    normalized_predictions, 
    normalized_future_predictions, 
    y_test, 
    scaler, 
    ticker,
    validation_indices=None,
    window_size=60
):
    """
    Create a comprehensive plot of actual vs predicted stock prices
    
    Args:
        stock_data (pd.DataFrame): Original stock price data
        normalized_predictions (np.ndarray): Normalized test predictions
        normalized_future_predictions (np.ndarray): Normalized future predictions
        y_test (np.ndarray): Actual test target values
        scaler (MinMaxScaler): Scaler used for normalization
        ticker (str): Stock ticker symbol
        validation_indices (list, optional): Indices for validation windows
        window_size (int): Size of the input window
    
    Returns:
        matplotlib.figure.Figure: Prediction visualization plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Ensure inputs are numpy arrays
    normalized_predictions = np.array(normalized_predictions).flatten()
    normalized_future_predictions = np.array(normalized_future_predictions).flatten()
    y_test = np.array(y_test).flatten()
    
    # Convert validation_indices to list if it's not already
    if validation_indices is not None and not isinstance(validation_indices, list):
        validation_indices = list(validation_indices)
    
    # Extract dates and prices from stock data
    stock_dates = np.array(stock_data.index)
    stock_prices = stock_data['Close'].values
    
    # Rescale predictions
    predictions_rescaled = scaler.inverse_transform(normalized_predictions.reshape(-1, 1)).flatten()
    future_predictions_rescaled = scaler.inverse_transform(normalized_future_predictions.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # Calculate first prediction target date
    if validation_indices and len(validation_indices) > 0:
        # Get the first day of the validation period
        first_validation_index = validation_indices[0]
        first_validation_date = stock_dates[first_validation_index]
        
        # Filter stock data from the first validation date
        mask = stock_dates >= first_validation_date
        filtered_stock_dates = stock_dates[mask]
        filtered_stock_prices = stock_prices[mask]
        
        # Top subplot: Full price chart with predictions
        ax1.plot(filtered_stock_dates, filtered_stock_prices, label='Actual Price', color='blue', linewidth=2)
        
        # Prepare prediction dates and values
        prediction_dates = []
        validation_predictions = []
        validation_y_test = []
        
        # Iterate through predictions to get dates and values
        for i in range(len(normalized_predictions)):
            # Get the prediction date 
            pred_date = stock_dates[first_validation_index + i + window_size - 1]
            
            # Only add if date is in validation period
            if pred_date >= first_validation_date:
                prediction_dates.append(pred_date)
                validation_predictions.append(predictions_rescaled[i])
                validation_y_test.append(y_test_rescaled[i])
        
        # Convert to numpy arrays
        prediction_dates = np.array(prediction_dates)
        validation_predictions_rescaled = np.array(validation_predictions)
        validation_y_test_rescaled = np.array(validation_y_test)
        
        # Plot validation predictions
        ax1.scatter(
            prediction_dates, 
            validation_predictions_rescaled, 
            color='red', 
            label='Validation Predictions', 
            marker='x', 
            s=100
        )
        
        # Plot actual validation prices
        ax1.scatter(
            prediction_dates, 
            validation_y_test_rescaled, 
            color='green', 
            label='Actual Validation Prices', 
            marker='o', 
            s=50, 
            alpha=0.7
        )
    
        # Future predictions
        if len(future_predictions_rescaled) > 0:
            # Calculate future prediction dates
            future_start_date = filtered_stock_dates[-1] + pd.Timedelta(days=1)
            future_dates = pd.date_range(
                start=future_start_date, 
                periods=len(future_predictions_rescaled)
            )
            
            ax1.scatter(
                future_dates, 
                future_predictions_rescaled, 
                color='purple', 
                label='Future Predictions (Next 30 Days)', 
                marker='x', 
                s=100
            )
            
            # Annotate the start and end of future predictions
            ax1.annotate(
                f'Future Prediction Start: {future_dates[0].date()}', 
                xy=(future_dates[0], future_predictions_rescaled[0]), 
                xytext=(10, 10), 
                textcoords='offset points', 
                ha='left', 
                va='bottom',
                fontsize=9,
                color='purple'
            )
            ax1.annotate(
                f'Future Prediction End: {future_dates[-1].date()}', 
                xy=(future_dates[-1], future_predictions_rescaled[-1]), 
                xytext=(10, -10), 
                textcoords='offset points', 
                ha='left', 
                va='top',
                fontsize=9,
                color='purple'
            )
        
        ax1.set_title(f'{ticker} Stock Price: Actual vs Predicted', fontsize=16)
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Bottom subplot: Prediction Errors
        # Calculate prediction errors
        prediction_errors = np.abs(validation_y_test_rescaled - validation_predictions_rescaled)
        
        # Plot prediction errors
        ax2.bar(
            prediction_dates, 
            prediction_errors, 
            color='orange', 
            alpha=0.6, 
            label='Prediction Errors'
        )
        
        ax2.set_title('Prediction Errors', fontsize=16)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Absolute Error ($)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate and align the tick labels
    plt.gcf().autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

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
