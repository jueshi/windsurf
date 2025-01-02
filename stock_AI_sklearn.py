"""
Change Log:
----------
2025-01-01:
- Created scikit-learn version of stock prediction script
- Replaced TensorFlow/Keras neural networks with ensemble methods (RandomForest, GradientBoosting, ExtraTrees)
- Simplified sequence creation and feature scaling
- Maintained core technical indicator calculations
- Implemented ensemble prediction using model averaging
- Added performance metrics (MSE, MAE, R2 Score)

"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Ensure stock_data directory exists
STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'stock_data')
os.makedirs(STOCK_DATA_DIR, exist_ok=True)

def is_market_holiday(date):
    """
    Check if a given date is a market holiday.
    
    Args:
        date (datetime or str): Date to check
    
    Returns:
        bool: True if the date is a market holiday, False otherwise
    """
    try:
        # Use NYSE calendar
        nyse = mcal.get_calendar('NYSE')
        
        # Convert input to pandas Timestamp if it's not already
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Get market holidays for the year
        market_holidays = nyse.holidays
        
        # Check if the date is in market holidays
        # Handle different possible return types from holidays
        if hasattr(market_holidays, 'date'):
            is_holiday = date.date() in market_holidays.date
        elif isinstance(market_holidays, pd.DatetimeIndex):
            is_holiday = date in market_holidays
        elif hasattr(market_holidays, '__iter__'):
            is_holiday = any(date.date() == h.date() for h in market_holidays)
        else:
            # Fallback method
            is_holiday = False
        
        # Additional checks for weekend
        is_weekend = date.dayofweek >= 5  # 5 and 6 are Saturday and Sunday
        
        return is_holiday or is_weekend
    except Exception as e:
        print(f"Error checking market holiday: {e}")
        # If there's an error, assume it's not a holiday
        return False

def download_stock_data(ticker, start_date, end_date):
    """
    Download or load stock data with local caching.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str or datetime): Start date for data download
        end_date (str or datetime): End date for data download
    
    Returns:
        pd.DataFrame or None: Stock data, or None if no data is available
    """
    # Convert dates to pandas Timestamp
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    
    # Check if end date is a market holiday or weekend
    if is_market_holiday(end_date_ts):
        print(f"Skipping data download for {ticker}: {end_date} is a market holiday or weekend")
        
        # Try to load existing local data
        local_filename = os.path.join(STOCK_DATA_DIR, f'{ticker}_{start_date}_to_{end_date}.csv')
        if os.path.exists(local_filename):
            print(f"Loading existing local data for {ticker}")
            return pd.read_csv(local_filename, parse_dates=['Date'], index_col='Date')
        
        return None
    
    # Create local filename
    local_filename = os.path.join(STOCK_DATA_DIR, f'{ticker}_{start_date}_to_{end_date}.csv')
    
    # Check if local file exists and is up to date
    if os.path.exists(local_filename):
        local_data = pd.read_csv(local_filename, parse_dates=['Date'], index_col='Date')
        
        # Check if local data covers the entire date range
        if (local_data.index.min().date() <= start_date_ts.date() and 
            local_data.index.max().date() >= end_date_ts.date()):
            print(f"Using local data for {ticker}")
            return local_data
    
    # Download data if no local file or incomplete data
    try:
        print(f"Downloading data for {ticker}")
        stock_data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            progress=False
        )
        
        # Additional validation
        if stock_data is None or stock_data.empty:
            print(f"No data downloaded for {ticker}")
            return None
        
        # Save to local file
        stock_data.reset_index(inplace=True)
        stock_data.to_csv(local_filename, index=False)
        
        return stock_data
    
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        
        # Fallback to local file if download fails
        if os.path.exists(local_filename):
            print(f"Falling back to local data for {ticker}")
            return pd.read_csv(local_filename, parse_dates=['Date'], index_col='Date')
        
        return None

# Technical Indicators (similar to original script)
def calculate_technical_indicators(df):
    """Calculate technical indicators for stock data."""
    # RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    macd, signal, hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    
    return df

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    middle_band = data.rolling(window=window).mean()
    std_dev = data.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def create_sequences(data, close_prices, look_back=90, prediction_days=5):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - look_back - prediction_days + 1):
        # Flatten features and include full look_back window
        X.append(data[i:i+look_back].flatten())
        # Target is the future price change percentage
        future_prices = close_prices[i+look_back:i+look_back+prediction_days]
        price_change_pct = (future_prices[-1] - future_prices[0]) / future_prices[0] * 100
        y.append(price_change_pct)
    return np.array(X), np.array(y)

def process_stock_data(stock_data, look_back=90, prediction_days=5):
    """Process stock data for machine learning model."""
    # Add technical indicators
    stock_data = calculate_technical_indicators(stock_data)
    
    # Select features
    features = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                'BB_Upper', 'BB_Middle', 'BB_Lower', 'Volume']
    
    # Drop rows with NaN
    stock_data = stock_data.dropna()
    
    # Prepare data for sequence creation
    data = stock_data[features].values
    close_prices = stock_data['Close'].values
    
    # Create sequences
    X, y = create_sequences(data, close_prices, look_back, prediction_days)
    
    # Scale features
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, {'X': scaler_X, 'y': scaler_y, 'close_prices': close_prices}

def train_ensemble_model(X_train, y_train, X_val, y_val, num_models=3):
    """Train an ensemble of scikit-learn models."""
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        ExtraTreesRegressor(n_estimators=100, random_state=42)
    ]
    
    trained_models = []
    for model in models[:num_models]:
        model.fit(X_train, y_train)
        trained_models.append(model)
    
    return trained_models

def advanced_ensemble_prediction(X_val, y_val, models, scalers):
    """
    Make predictions using ensemble of models.
    
    Args:
        X_val (np.array): Validation input data
        y_val (np.array): Validation target data
        models (list): List of trained models
        scalers (dict): Dictionary of scalers used for each feature
    
    Returns:
        tuple: Predictions for next 5 days, actual values, and statistical summary
    """
    # Predict with each model
    predictions = [model.predict(X_val) for model in models]
    
    # Average predictions
    mean_prediction = np.mean(predictions, axis=0)
    
    # Inverse transform predictions and actual values
    mean_prediction_inv = scalers['y'].inverse_transform(mean_prediction.reshape(-1, 1)).ravel()
    y_val_inv = scalers['y'].inverse_transform(y_val.reshape(-1, 1)).ravel()
    
    # Calculate prediction statistics
    mse = np.mean((y_val_inv - mean_prediction_inv) ** 2)
    mae = np.mean(np.abs(y_val_inv - mean_prediction_inv))
    r2 = 1 - (np.sum((y_val_inv - mean_prediction_inv) ** 2) / np.sum((y_val_inv - np.mean(y_val_inv)) ** 2))
    
    # Reconstruct actual and predicted prices
    close_prices = scalers['close_prices']
    
    # Predict next 5 days for each validation point
    actual_prices = []
    predicted_prices = []
    
    for i in range(len(y_val_inv)):
        # Use the last known price as the base
        base_price = close_prices[-(len(y_val_inv)-i)]
        
        # Predict price change percentage
        price_change_pct = mean_prediction_inv[i]
        
        # Calculate predicted and actual prices
        actual_price = base_price * (1 + y_val_inv[i]/100)
        predicted_price = base_price * (1 + price_change_pct/100)
        
        actual_prices.append(actual_price)
        predicted_prices.append(predicted_price)
    
    # Prepare detailed prediction results
    prediction_details = {
        'base_prices': close_prices[-(len(y_val_inv)):],
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'price_change_pct': mean_prediction_inv
    }
    
    stats = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }
    
    return prediction_details, stats

def predict_next_5_days(X_val, models, scalers):
    """
    Predict next 5 days using ensemble of models.
    
    Args:
        X_val (np.array): Validation input data
        models (list): List of trained models
        scalers (dict): Dictionary of scalers used for each feature
    
    Returns:
        dict: Predictions for next 5 days
    """
    # Predict with each model
    predictions = [model.predict(X_val) for model in models]
    
    # Average predictions
    mean_prediction = np.mean(predictions, axis=0)
    
    # Inverse transform predictions
    mean_prediction_inv = scalers['y'].inverse_transform(mean_prediction.reshape(-1, 1)).ravel()
    
    # Reconstruct predicted prices
    close_prices = scalers['close_prices']
    
    # Predict next 5 days
    predicted_prices = []
    base_price = close_prices[-1]
    
    for i in range(5):
        # Predict price change percentage
        price_change_pct = mean_prediction_inv[i]
        
        # Calculate predicted price
        predicted_price = base_price * (1 + price_change_pct/100)
        
        predicted_prices.append(predicted_price)
    
    # Prepare detailed prediction results
    next_5_days_prediction = {
        'base_prices': [base_price] * 5,
        'predicted_prices': predicted_prices,
        'price_change_pcts': mean_prediction_inv[:5]
    }
    
    return next_5_days_prediction

def plot_stock_predictions(prediction_details, next_5_days_prediction, ticker):
    """
    Plot actual vs predicted stock prices for validation period and next 5 days.
    
    Args:
        prediction_details (dict): Dictionary containing validation period prediction details
        next_5_days_prediction (dict): Dictionary containing next 5 days prediction details
        ticker (str): Stock ticker symbol
    """
    plt.figure(figsize=(20, 10))
    
    # Validation Period Plotting
    actual_prices = prediction_details['actual_prices']
    predicted_prices = prediction_details['predicted_prices']
    
    # Plot actual prices
    plt.plot(range(len(actual_prices)), actual_prices, label='Actual Prices (Validation)', color='blue', marker='o')
    
    # Plot predicted prices
    plt.plot(range(len(predicted_prices)), predicted_prices, label='Predicted Prices (Validation)', color='red', linestyle='--', marker='x')
    
    # Highlight 5-day prediction windows during validation
    window_size = 5
    for i in range(len(actual_prices) - window_size + 1):
        window_actual = actual_prices[i:i+window_size]
        window_predicted = predicted_prices[i:i+window_size]
        
        # Plot 5-day window predictions
        plt.plot(
            range(i, i+window_size), 
            window_predicted, 
            color='green', 
            linestyle=':', 
            marker='^', 
            alpha=0.5,
            label='5-Day Prediction Window' if i == 0 else ""
        )
    
    # Next 5 Days Prediction Plotting
    # Create x-axis for next 5 days (continuing from validation period)
    next_5_days_x = list(range(len(actual_prices), len(actual_prices) + 5))
    
    # Plot predicted prices for next 5 days
    plt.plot(next_5_days_x, next_5_days_prediction['predicted_prices'], 
             label='Predicted Prices (Next 5 Days)', color='purple', linestyle='--', marker='^')
    
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{ticker}_stock_prediction_without_base_price.png', dpi=300)
    plt.close()

def predict_stock_prices(tickers, start_date, end_date):
    """Predict stock prices for given tickers."""
    prediction_results = {}
    
    for ticker in tickers:
        try:
            # Download stock data
            stock_data = download_stock_data(ticker, start_date, end_date)
            
            # Skip if no data is available
            if stock_data is None:
                print(f"Skipping {ticker} due to no available data")
                continue
            
            # Process stock data
            X, y, scalers = process_stock_data(stock_data)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train ensemble models
            models = train_ensemble_model(X_train, y_train, X_val, y_val)
            
            # Make predictions for validation period
            prediction_details, stats = advanced_ensemble_prediction(X_val, y_val, models, scalers)
            
            # Predict next 5 days
            next_5_days_prediction = predict_next_5_days(X_val, models, scalers)
            
            # Plot predictions
            plot_stock_predictions(prediction_details, next_5_days_prediction, ticker)
            
            # Print prediction results
            print(f"\nPrediction Results for {ticker}:")
            print("Prediction Statistics:")
            for stat, value in stats.items():
                print(f"{stat}: {value}")
            
            # Print detailed price predictions for validation period
            print("\nDetailed Price Predictions (Validation Period):")
            for i in range(len(prediction_details['base_prices'])):
                print(f"Day {i+1}:")
                print(f"  Base Price: ${float(prediction_details['base_prices'][i]):.2f}")
                print(f"  Actual Price: ${float(prediction_details['actual_prices'][i]):.2f}")
                print(f"  Predicted Price: ${float(prediction_details['predicted_prices'][i]):.2f}")
                print(f"  Price Change %: {float(prediction_details['price_change_pct'][i]):.2f}%")
            
            # Print next 5 days predictions
            print("\nNext 5 Days Predictions:")
            for i in range(5):
                print(f"Day {i+1}:")
                print(f"  Base Price: ${float(next_5_days_prediction['base_prices'][i]):.2f}")
                print(f"  Predicted Price: ${float(next_5_days_prediction['predicted_prices'][i]):.2f}")
                print(f"  Price Change %: {float(next_5_days_prediction['price_change_pcts'][i]):.2f}%")
            
            # Store results
            prediction_results[ticker] = {
                'validation_period': prediction_details,
                'next_5_days': next_5_days_prediction
            }
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    return prediction_results

# Main execution
if __name__ == '__main__':
    # Define tickers to predict
    tickers_to_predict = ['SPY']
    
    # Specify date range
    start_date = '2020-01-02'
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # Run prediction workflow
    prediction_results = predict_stock_prices(
        tickers=tickers_to_predict, 
        start_date=start_date, 
        end_date=end_date
    )
