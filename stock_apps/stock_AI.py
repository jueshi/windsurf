"""
Change Log:
----------
2024-12-31 (2):
- Replaced external technical indicator library with custom implementations
- Added robust fallback for technical indicators
- Removed external library dependencies
- Maintained feature engineering complexity

2024-12-31:
- Enhanced AI model:
  - Added technical indicators (RSI, MACD, Bollinger Bands)
  - Implemented residual connections
  - Added L1/L2 regularization
  - Increased model capacity
  - Added batch normalization
  - Improved feature engineering

2024-12-30 (2):
- Enhanced model to better predict drawdowns:
  - Added rolling volatility and drawdown features
  - Increased lookback period to 20 days
  - Added dropout layers to prevent overfitting
  - Added bidirectional LSTM for better pattern recognition
  - Added attention mechanism for key pattern detection

2024-12-30:
- Added rolling predictions for validation period
- Parameterized prediction days (PREDICTION_DAYS)
- Added visualization of prediction bands
- Reduced prediction horizon to 5 days
- Added error analysis for each future day
- Improved plot formatting and readability
- Added configurable plot density (PLOT_EVERY_N_PREDICTIONS)

2024-12-29:
- Initial implementation of stock price prediction
- Added LSTM model for time series forecasting
- Implemented log transformation of prices
- Added basic error handling and data validation
- Created multi-ticker support with subplot grid

2025-01-01:
- Enhanced model with advanced technical indicators, ensemble learning, and improved sequence creation

2025-01-02:
- Added comprehensive prediction visualization and error analysis function

2025-01-03:
- Added robust data download and local caching mechanism

2025-01-04:
- Updated main prediction workflow with comprehensive error handling and flexibility

2025-01-05:
- Updated plot_predictions_vs_actual to handle multi-dimensional arrays and improve plotting

2025-01-06:
- Updated advanced_ensemble_prediction to handle percentage changes and convert back to absolute prices

2025-01-07:
- Improved data preprocessing in process_stock_data to handle outliers

2025-01-08:
- Updated create_advanced_sequences to use percentage changes for more stable predictions

2025-01-09:
- Added comprehensive error handling to advanced_ensemble_prediction to prevent NaN values

2025-01-10:
- Added comprehensive error handling to plot_predictions_vs_actual to prevent crashes with invalid input

2025-01-11:
- Added comprehensive diagnostic logging to create_advanced_sequences

2025-01-12:
- Added market holiday check to data download function

2025-01-13:
- Improved market holiday check to include weekend detection and handle different holiday return types

2025-01-14:
- Updated scaling to store more information about scalers and handle them more robustly

2025-01-15:
- Updated evaluation to handle different scaler input types more robustly
"""

import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Dropout, BatchNormalization, 
    Conv1D, MaxPooling1D, Bidirectional, 
    LeakyReLU, Add, Activation, LayerNormalization,
    MultiHeadAttention
)
from tensorflow.keras.regularizers import l1_l2, l2
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import timedelta
import pandas_market_calendars as mcal
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Configuration parameters
LOOK_BACK = 90  # Increased look-back period
PREDICTION_DAYS = 5  # Reduced prediction horizon
ENSEMBLE_MODELS = 3  # Number of models in ensemble
PLOT_EVERY_N_PREDICTIONS = 2  # Plot every Nth prediction to avoid overcrowding

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI) manually.
    
    Args:
        data (pd.Series): Price data
        window (int): Lookback period for RSI calculation
    
    Returns:
        pd.Series: RSI values
    """
    # Ensure data is numeric
    data = pd.to_numeric(data, errors='coerce')
    
    # Drop NaN values
    data = data.dropna()
    
    # Check if we have enough data
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)
    
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window, min_periods=1).mean()
    avg_loss = losses.rolling(window=window, min_periods=1).mean()
    
    # Calculate relative strength
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    # Exponential Moving Averages
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    
    # MACD Line
    macd = exp1 - exp2
    
    # Signal Line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    # MACD Histogram
    macd_hist = macd - signal_line
    
    return macd, signal_line, macd_hist

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Bollinger Band Percentage
    bb_pct = (data - lower_band) / (upper_band - lower_band)
    
    return upper_band, lower_band, bb_pct

def calculate_technical_indicators(df):
    """
    Calculate advanced technical indicators for stock data.
    
    Args:
        df (pd.DataFrame): Input stock price data
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Ensure input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Input must be a pandas DataFrame")
        return df
    
    # Handle multi-level column names
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-level column names
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Print column names for debugging
    print(f"Available columns: {list(df.columns)}")
    
    # Handle special file structure with headers and ticker name
    if len(df) > 2 and all(isinstance(col, str) for col in df.columns):
        # Assume first row is headers, second row is ticker, third row is data
        try:
            # Use the third row onwards as actual data
            df = df.iloc[2:].reset_index(drop=True)
            
            # Rename columns if needed
            df.columns = [col.strip() for col in df.columns]
        except Exception as e:
            print(f"Error processing multi-row header: {e}")
    
    # Specific column mapping based on the given order: date, close, high, low, open, volume
    column_mapping = {
        0: 'Date',
        1: 'Close',
        2: 'High', 
        3: 'Low',
        4: 'Open',
        5: 'Volume'
    }
    
    # If columns are in numeric index order, rename them
    if all(isinstance(col, int) for col in df.columns):
        df.columns = [column_mapping.get(i, f'Column_{i}') for i in df.columns]
    else:
        # Standardize column names for string-based columns
        column_mapping_str = {
            'date': 'Date',
            'close': 'Close',
            'high': 'High',
            'low': 'Low', 
            'open': 'Open',
            'volume': 'Volume'
        }
        
        # Handle both single and multi-level column names
        df.columns = [
            column_mapping_str.get(
                col.lower() if isinstance(col, str) else col[0].lower(), 
                col
            ) 
            for col in df.columns
        ]
    
    # Ensure Date column is parsed if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    
    # Ensure required columns exist
    required_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    
    # Check which required columns are actually present
    available_columns = [col for col in required_columns if col in df.columns]
    
    if not available_columns:
        print(f"No valid columns found. Available columns: {list(df.columns)}")
        return df
    
    # Convert columns to numeric, coercing errors
    for col in available_columns:
        try:
            # First, try to convert the entire column
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except TypeError:
            # If that fails, convert column values one by one
            df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    
    # Only drop rows with NaN in available columns
    df.dropna(subset=available_columns, inplace=True)
    
    # Verify we still have data
    if len(df) == 0:
        print("No valid data remaining after cleaning")
        return pd.DataFrame()
    
    # Rest of the technical indicators calculation remains the same
    try:
        # Moving Averages
        if 'Close' in df.columns:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential Moving Averages
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'Close' in df.columns:
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # MACD
        if 'Close' in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Stochastic Oscillator
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['%K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
            df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ['High', 'Low', 'Close']):
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = tr.ewm(span=14).mean()
        
        # On-Balance Volume (OBV)
        if all(col in df.columns for col in ['Close', 'Volume']):
            obv = [0]
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv.append(obv[-1] + df['Volume'].iloc[i])
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv.append(obv[-1] - df['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['OBV'] = obv
    
    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
    
    return df

def calculate_features(df):
    """
    Calculate advanced technical indicators with robust normalization and range control.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Close' price column
    
    Returns:
        pd.DataFrame: DataFrame with normalized and range-controlled technical features
    """
    # Ensure Close price is present and sorted
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' price column")
    
    # Compute percentage change first
    df['pct_change'] = df['Close'].pct_change()
    
    # Advanced normalization function with range limiting
    def robust_normalize(series, clip_std=3):
        """
        Normalize series with robust clipping to control extreme values.
        
        Args:
            series (pd.Series): Input series to normalize
            clip_std (float): Number of standard deviations to clip values
        
        Returns:
            pd.Series: Normalized and clipped series
        """
        # Calculate mean and standard deviation
        mean = series.mean()
        std = series.std()
        
        # Normalize
        normalized = (series - mean) / (std + 1e-8)
        
        # Clip extreme values
        clipped = np.clip(normalized, -clip_std, clip_std)
        
        return clipped
    
    # Normalize Close price
    df['Close_normalized'] = robust_normalize(df['Close'])
    
    # Volatility calculations with robust normalization
    df['volatility'] = robust_normalize(df['Close'].rolling(window=5).std())
    
    # Momentum calculations
    df['momentum_1d'] = robust_normalize(df['pct_change'])
    df['momentum_5d'] = robust_normalize(df['Close'].pct_change(5))
    df['momentum_20d'] = robust_normalize(df['Close'].pct_change(20))
    
    # Advanced volatility metrics
    df['volatility_5d'] = robust_normalize(df['Close'].rolling(window=5).std())
    df['volatility_ratio'] = robust_normalize(
        df['volatility_5d'] / (df['volatility_5d'].std() + 1e-8)
    )
    
    # Drawdown calculations with robust normalization
    def calculate_drawdown(close_series, window):
        rolling_max = close_series.rolling(window=window, min_periods=1).max()
        drawdown = (close_series - rolling_max) / (rolling_max + 1e-8)
        return robust_normalize(drawdown)
    
    df['drawdown'] = calculate_drawdown(df['Close'], 20)
    df['drawdown_5d'] = calculate_drawdown(df['Close'], 5)
    
    # Return distribution statistics
    df['return_skew'] = robust_normalize(
        df['pct_change'].rolling(window=20).apply(lambda x: x.skew())
    )
    df['return_kurt'] = robust_normalize(
        df['pct_change'].rolling(window=20).apply(lambda x: x.kurtosis())
    )
    
    # Price level comparisons
    sma_20 = df['Close'].rolling(window=20).mean()
    sma_5 = df['Close'].rolling(window=5).mean()
    
    # Ensure single-column operations
    df['price_sma_20'] = robust_normalize(sma_20)
    df['price_sma_5'] = robust_normalize(sma_5)
    
    # Careful price ratio calculation
    df['price_ratio'] = robust_normalize(
        df['Close'] / (sma_20 + 1e-8)
    )
    
    # RSI Calculation with robust normalization
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi'] = robust_normalize(100 - (100 / (1 + rs)))
    
    # MACD Calculation
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = robust_normalize(exp1 - exp2)
    df['macd_signal'] = robust_normalize(
        df['macd'].ewm(span=9, adjust=False).mean()
    )
    df['macd_diff'] = robust_normalize(df['macd'] - df['macd_signal'])
    
    # Bollinger Bands with careful calculation
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    
    # Compute Bollinger Bands components
    bb_high = rolling_mean + (rolling_std * 2)
    bb_low = rolling_mean - (rolling_std * 2)
    
    # Careful Bollinger Bands percentage calculation
    bb_pct_series = (df['Close'] - bb_low) / (bb_high - bb_low + 1e-8)
    
    # Normalize and store as single column
    df['bb_high'] = robust_normalize(bb_high)
    df['bb_low'] = robust_normalize(bb_low)
    df['bb_pct'] = robust_normalize(bb_pct_series)
    
    # Drop initial NaN rows
    df.dropna(inplace=True)
    
    return df

def calculate_advanced_indicators(df):
    """
    Calculate more sophisticated technical indicators.
    
    Args:
        df (pd.DataFrame): Input price data
    
    Returns:
        pd.DataFrame: DataFrame with additional technical features
    """
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Ensure we have the correct column names for a MultiIndex DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        # Flatten MultiIndex columns
        data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
    
    # Ensure we have the correct columns
    required_columns = ['Close_SPY', 'High_SPY', 'Low_SPY']
    if not all(col in data.columns for col in required_columns):
        print("Missing required columns:", 
              [col for col in required_columns if col not in data.columns])
        return df
    
    # Extract Close, High, Low columns
    close = data['Close_SPY']
    high = data['High_SPY']
    low = data['Low_SPY']
    
    # Existing indicators
    data['pct_change'] = close.pct_change()
    data['log_return'] = np.log(close / close.shift(1))
    
    # Advanced moving averages
    data['sma_20'] = close.rolling(window=20).mean()
    data['sma_50'] = close.rolling(window=50).mean()
    data['ema_20'] = close.ewm(span=20, adjust=False).mean()
    
    # Volatility measures
    data['rolling_std_20'] = close.rolling(window=20).std()
    data['atr_14'] = calculate_atr(high, low, close)
    
    # Momentum indicators
    data['rsi_14'] = calculate_rsi(close)
    data['macd'], data['signal'], data['hist'] = calculate_macd(close)
    
    # Advanced oscillators
    data['stoch_k'], data['stoch_d'] = calculate_stochastic(close)
    
    # Trend indicators
    data['adx'] = calculate_rsi(close - close.rolling(window=14).mean())
    
    # Normalize and fill NaNs
    columns_to_normalize = ['pct_change', 'log_return', 'sma_20', 'sma_50', 
                            'ema_20', 'rolling_std_20', 'atr_14', 
                            'rsi_14', 'macd', 'signal', 'hist',
                            'stoch_k', 'stoch_d', 'adx']
    
    scaler = MinMaxScaler()
    data[columns_to_normalize] = data[columns_to_normalize].apply(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
    )
    
    # Fill NaNs
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    
    # Add new features back to original DataFrame
    for col in data.columns:
        if col not in df.columns and col not in ['Close_SPY', 'High_SPY', 'Low_SPY']:
            df[col] = data[col]
    
    return df

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) manually.
    
    Args:
        data (pd.Series): Price data
        fast_period (int): Fast moving average period
        slow_period (int): Slow moving average period
        signal_period (int): Signal line period
    
    Returns:
        tuple: MACD, Signal Line, MACD Histogram
    """
    # Calculate exponential moving averages
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd = fast_ema - slow_ema
    
    # Calculate signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    histogram = macd - signal
    
    return macd, signal, histogram

def calculate_stochastic(high, low, close, window=14, smooth_window=3):
    """
    Calculate Stochastic Oscillator manually.
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Closing prices
        window (int): Lookback period
        smooth_window (int): Smoothing period
    
    Returns:
        tuple: Stochastic %K, Stochastic %D
    """
    # Ensure inputs are numeric
    high = pd.to_numeric(high, errors='coerce')
    low = pd.to_numeric(low, errors='coerce')
    close = pd.to_numeric(close, errors='coerce')
    
    # Drop NaN values
    data = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    }).dropna()
    
    # Validate window parameters
    window = max(1, int(window))
    smooth_window = max(1, int(smooth_window))
    
    # Check if we have enough data
    if len(data) < window:
        return (
            pd.Series([np.nan] * len(data), index=data.index),
            pd.Series([np.nan] * len(data), index=data.index)
        )
    
    # Calculate lowest low and highest high in the window
    lowest_low = data['low'].rolling(window=window, min_periods=1).min()
    highest_high = data['high'].rolling(window=window, min_periods=1).max()
    
    # Calculate %K (Stochastic Oscillator)
    stochastic_k = 100 * (data['close'] - lowest_low) / (highest_high - lowest_low)
    
    # Smooth %K to get %D
    stochastic_d = stochastic_k.rolling(window=smooth_window, min_periods=1).mean()
    
    return stochastic_k, stochastic_d

def calculate_atr(high, low, close, window=14):
    """
    Calculate Average True Range (ATR) manually.
    
    Args:
        high (pd.Series): High prices
        low (pd.Series): Low prices
        close (pd.Series): Closing prices
        window (int): Lookback period
    
    Returns:
        pd.Series: Average True Range
    """
    # Calculate True Range
    high_low = high - low
    high_close_prev = np.abs(high - close.shift())
    low_close_prev = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, high_close_prev, low_close_prev)
    
    # Calculate ATR
    atr = true_range.ewm(span=window, adjust=False).mean()
    
    return atr

def create_sequences(data, look_back=LOOK_BACK):
    """
    Create sequences with multiple features and robust error handling.
    
    Args:
        data (pd.DataFrame): Input DataFrame with features
        look_back (int): Number of historical days to use for each sequence
    
    Returns:
        tuple: Numpy arrays of input sequences (X) and target values (y)
    """
    # Predefined feature columns
    feature_cols = ['pct_change', 'Close_normalized', 'volatility', 'momentum_1d', 'momentum_5d', 
                    'momentum_20d', 'volatility_5d', 'volatility_ratio', 
                    'drawdown', 'drawdown_5d', 'return_skew', 'return_kurt', 
                    'price_sma_20', 'price_sma_5', 'price_ratio', 
                    'rsi', 'macd', 'macd_signal', 'macd_diff', 
                    'bb_high', 'bb_low', 'bb_pct']
    
    # Validate input data
    if len(data) < look_back + PREDICTION_DAYS + 10:
        print(f"ERROR: Insufficient data for sequence creation. Need at least {look_back + PREDICTION_DAYS + 10} points.")
        return np.array([]), np.array([])
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return np.array([]), np.array([])
    
    # Prepare containers
    X, y = [], []
    
    # Detailed logging
    invalid_sequences = {
        'nan_values': 0,
        'inf_values': 0,
        'out_of_range': 0,
        'shape_mismatch': 0
    }
    
    # Determine maximum possible sequences
    max_sequences = len(data) - look_back - PREDICTION_DAYS
    
    # Sequence creation with extensive error checking
    valid_sequences = 0
    for i in range(max_sequences):
        try:
            # Extract sequence
            sequence = data.iloc[i:i+look_back][feature_cols].values
            
            # Validate sequence shape
            if sequence.shape[0] != look_back:
                invalid_sequences['shape_mismatch'] += 1
                continue
            
            # Check for NaN values
            if np.isnan(sequence).any():
                invalid_sequences['nan_values'] += 1
                continue
            
            # Check for infinite values
            if np.isinf(sequence).any():
                invalid_sequences['inf_values'] += 1
                continue
            
            # Check for extreme values (beyond 5 standard deviations)
            if np.any(np.abs(sequence) > 5):
                invalid_sequences['out_of_range'] += 1
                continue
            
            # Get future percentage changes
            target = data.iloc[i+look_back:i+look_back+PREDICTION_DAYS]['pct_change'].values
            
            # Validate target
            if len(target) != PREDICTION_DAYS:
                continue
            
            # Check for NaN or infinite values in target
            if (np.isnan(target).any() or 
                np.isinf(target).any() or 
                np.any(np.abs(target) > 0.2)):  # Limit to 20% daily change
                continue
            
            X.append(sequence)
            y.append(target)
            valid_sequences += 1
            
            # Optional: limit total sequences to prevent memory issues
            if valid_sequences >= 500:
                break
        
        except Exception as e:
            print(f"Unexpected error processing sequence at index {i}: {e}")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Detailed logging of sequence generation
    print("\nSequence Creation Diagnostic Report:")
    print(f"Total potential sequences: {max_sequences}")
    print(f"Generated valid sequences: {len(X)}")
    print("Invalid Sequence Breakdown:")
    for reason, count in invalid_sequences.items():
        print(f"  - {reason}: {count}")
    
    print(f"\nFinal Sequence Details:")
    print(f"Sequence shape: {X.shape if len(X) > 0 else 'N/A'}")
    print(f"Target shape: {y.shape if len(y) > 0 else 'N/A'}")
    
    # Additional sanity checks
    if len(X) == 0:
        print("WARNING: No valid sequences generated. Check data preprocessing.")
    
    return X, y

def create_advanced_sequences(data, look_back=90, prediction_days=5):
    """
    Create more robust sequences with multiple features and advanced error handling.
    
    Args:
        data (np.ndarray or pd.DataFrame): Input data with features
        look_back (int): Number of previous days to use for prediction
        prediction_days (int): Number of days to predict
    
    Returns:
        tuple: Numpy arrays of input sequences (X) and target values (y)
    """
    # Diagnostic print for input data
    print(f"Input data shape: {data.shape}")
    print(f"Input data type: {type(data)}")
    print(f"Input data dtype: {data.dtype}")
    print(f"Input data NaN check: {np.isnan(data).any()}")
    print(f"Input data infinite check: {np.isinf(data).any()}")
    
    # Convert to numpy array if not already
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    # Validate input
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a numpy array or pandas DataFrame")
    
    # Ensure 2D array
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    # Check if we have enough data
    if len(data) < look_back + prediction_days:
        print(f"ERROR: Insufficient data: need at least {look_back + prediction_days} rows")
        print(f"Current data length: {len(data)}")
        raise ValueError(f"Insufficient data: need at least {look_back + prediction_days} rows")
    
    # Remove rows with NaN or infinite values
    original_rows = len(data)
    data = data[~np.isnan(data).any(axis=1)]
    data = data[~np.isinf(data).any(axis=1)]
    
    # Diagnostic print after NaN and infinite removal
    print(f"Rows removed due to NaN/infinite: {original_rows - len(data)}")
    print(f"Remaining data shape: {data.shape}")
    
    # Check data after cleaning
    if len(data) < look_back + prediction_days:
        print("ERROR: Insufficient valid data after removing NaNs and infinities")
        raise ValueError(f"Insufficient valid data after removing NaNs and infinities")
    
    # Prepare input sequences
    X, y = [], []
    
    # Sliding window to create sequences
    for i in range(len(data) - look_back - prediction_days + 1):
        # Input sequence: look_back days of data
        X_sequence = data[i:i+look_back]
        
        # Target: percentage changes for next prediction_days
        # Use the last column for target if multi-column, otherwise use the entire array
        if data.shape[1] > 1:
            base_price = data[i+look_back-1, -1]
            future_prices = data[i+look_back:i+look_back+prediction_days, -1]
        else:
            base_price = data[i+look_back-1, 0]
            future_prices = data[i+look_back:i+look_back+prediction_days, 0]
        
        # Skip sequences with invalid base price
        if base_price <= 0:
            continue
        
        # Calculate percentage changes
        pct_changes = (future_prices - base_price) / base_price
        
        # Clip extreme percentage changes to prevent scaling issues
        pct_changes = np.clip(pct_changes, 
                               np.percentile(pct_changes, 1), 
                               np.percentile(pct_changes, 99))
        
        # Ensure no NaN or infinite values in sequence
        if (np.isnan(X_sequence).any() or np.isinf(X_sequence).any() or 
            np.isnan(pct_changes).any() or np.isinf(pct_changes).any()):
            continue
        
        X.append(X_sequence)
        y.append(pct_changes)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Diagnostic print for final sequences
    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    print(f"X NaN check: {np.isnan(X).any()}")
    print(f"y NaN check: {np.isnan(y).any()}")
    print(f"X infinite check: {np.isinf(X).any()}")
    print(f"y infinite check: {np.isinf(y).any()}")
    
    # Final validation
    if len(X) == 0 or len(y) == 0:
        print("ERROR: No valid sequences could be created")
        raise ValueError("No valid sequences could be created")
    
    return X, y

def build_advanced_model(input_shape, output_days, learning_rate=0.001):
    """
    Build an advanced deep learning model for stock price prediction with enhanced architecture.
    
    Args:
        input_shape (tuple): Shape of input data
        output_days (int): Number of days to predict
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, LeakyReLU
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with more units and dropout
    x = LSTM(256, return_sequences=True, 
             kernel_regularizer=l2(0.001), 
             recurrent_regularizer=l2(0.001))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer
    x = LSTM(128, return_sequences=True, 
             kernel_regularizer=l2(0.001), 
             recurrent_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    
    # Third LSTM layer
    x = LSTM(64, return_sequences=False, 
             kernel_regularizer=l2(0.001), 
             recurrent_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    
    # Dense layers with ReLU and final output layer with ReLU activation
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
    
    # Output layer with ReLU to ensure non-negative predictions
    outputs = Dense(output_days, activation='relu', 
                   kernel_regularizer=l2(0.001), 
                   name='dense_output')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Custom loss function to penalize negative predictions
    def custom_mse_with_non_negativity(y_true, y_pred):
        # Mean Squared Error
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        
        # Penalty for negative predictions
        negative_penalty = tf.maximum(-y_pred, 0)
        penalty_loss = tf.reduce_mean(negative_penalty * 10)  # Adjust multiplier as needed
        
        return mse + penalty_loss
    
    # Compile model with custom loss and metrics
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=custom_mse_with_non_negativity,
        metrics=['mae', 'mse']
    )
    
    return model

def build_ensemble_model(input_shape, output_days):
    """
    Build an ensemble of neural network models with diverse architectures.
    
    Args:
        input_shape (tuple): Shape of input sequences
        output_days (int): Number of days to predict
    
    Returns:
        list: Ensemble of compiled models
    """
    models = []
    
    # Diverse model architectures with different regularization strategies
    for i in range(ENSEMBLE_MODELS):
        if i == 0:
            # Advanced regularized model
            model = build_advanced_model(input_shape, output_days)
        elif i == 1:
            # GRU-based model with different regularization
            model = Sequential([
                Bidirectional(GRU(
                    100, 
                    return_sequences=True, 
                    kernel_regularizer=l1(0.0001)
                ), input_shape=input_shape),
                BatchNormalization(),
                GlobalMaxPooling1D(),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dropout(0.3),
                Dense(output_days, kernel_regularizer=l2(0.0005))
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        else:
            # Simplified LSTM with different approach
            model = Sequential([
                LSTM(
                    256, 
                    input_shape=input_shape, 
                    kernel_regularizer=l1_l2(l1=0.00005, l2=0.00005)
                ),
                Dense(128, activation='relu', kernel_regularizer=l1(0.0001)),
                Dropout(0.4),
                Dense(64, activation='relu'),
                Dense(output_days)
            ])
            model.compile(
                optimizer=Adam(learning_rate=0.0008),
                loss='mse',
                metrics=['mae']
            )
        
        models.append(model)
    
    return models

class CyclicLR(tf.keras.callbacks.Callback):
    """
    Cyclic Learning Rate callback with robust implementation.
    
    Implements a cyclical learning rate policy that varies the learning rate
    between reasonable boundary values.
    
    Args:
        base_lr (float): Initial learning rate
        max_lr (float): Upper bound of learning rate
        step_size (int): Number of training iterations per half cycle
        mode (str): Learning rate policy ('triangular', 'triangular2', 'exp_range')
    """
    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000, mode='triangular2'):
        super().__init__()
        self.base_lr = float(base_lr)
        self.max_lr = float(max_lr)
        self.step_size = step_size
        self.mode = mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}
        
        # Compute cycle multiplier
        if self.mode == 'triangular2':
            self.cycle_multiplier = 1
        elif self.mode == 'exp_range':
            self.cycle_multiplier = 1
        else:
            self.cycle_multiplier = 1
    
    def clr(self):
        """
        Compute the learning rate based on the current iteration.
        
        Returns:
            float: Computed learning rate
        """
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.cycle_multiplier
        elif self.mode == 'exp_range':
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.exp(-x) * self.cycle_multiplier
        else:  # triangular
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        
        return float(lr)
    
    def _set_lr(self, lr):
        """
        Robust method to set learning rate across different optimizer types.
        
        Args:
            lr (float): Learning rate to set
        """
        try:
            # Detailed debugging of optimizer
            print(f"Optimizer type: {type(self.model.optimizer)}")
            print(f"Optimizer attributes: {dir(self.model.optimizer)}")
            
            # Multiple strategies to set learning rate
            if hasattr(self.model.optimizer, 'learning_rate'):
                if callable(self.model.optimizer.learning_rate):
                    # If it's a callable (like LearningRateSchedule)
                    self.model.optimizer.learning_rate.assign(lr)
                else:
                    # Direct attribute setting
                    self.model.optimizer.learning_rate = lr
            elif hasattr(self.model.optimizer, 'lr'):
                # Fallback to 'lr' attribute
                self.model.optimizer.lr = lr
            else:
                # Last resort: use backend
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            
            print(f"Successfully set learning rate to {lr}")
        except Exception as e:
            print(f"Comprehensive error setting learning rate: {e}")
            print(f"Optimizer details: {self.model.optimizer}")
            # Attempt to print more details about the optimizer
            try:
                print(f"Optimizer config: {self.model.optimizer.get_config()}")
            except Exception as config_error:
                print(f"Could not get optimizer config: {config_error}")
    
    def on_train_begin(self, logs=None):
        """Initialize learning rate at the start of training."""
        self._set_lr(self.base_lr)
    
    def on_batch_end(self, batch, logs=None):
        """Update learning rate at the end of each batch."""
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        
        lr = self.clr()
        self._set_lr(lr)
        
        self.history.setdefault('lr', []).append(lr)
        self.history.setdefault('iterations', []).append(self.trn_iterations)

def train_ensemble_model(X_train, y_train, X_val, y_val, scalers, num_models=5, epochs=100, batch_size=32):
    """
    Train an ensemble of advanced neural network models.
    
    Args:
        X_train (np.array): Training input features
        y_train (np.array): Training target values
        X_val (np.array): Validation input features
        y_val (np.array): Validation target values
        scalers (sklearn.preprocessing.RobustScaler): Feature scaler
        num_models (int, optional): Number of models in ensemble. Defaults to 5.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        batch_size (int, optional): Training batch size. Defaults to 32.
    
    Returns:
        tuple: Trained models and training histories
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Ensure y_train and y_val are 2D arrays
    y_train = np.atleast_2d(y_train)
    y_val = np.atleast_2d(y_val)
    
    # Callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-5
    )
    
    # Initialize lists to store models and histories
    models = []
    histories = []
    
    # Seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train multiple models
    for i in range(num_models):
        # Reset model for each iteration
        tf.keras.backend.clear_session()
        
        # Create model with slight variation
        model = build_advanced_model(
            input_shape=X_train.shape[1:], 
            output_days=y_train.shape[1], 
            learning_rate=np.random.uniform(0.0005, 0.002)
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Store model and history
        models.append(model)
        histories.append(history)
        
        print(f"\nModel {i+1} Training Complete")
    
    return models, histories

def preprocess_stock_data(df, look_back=90, prediction_days=5, test_size=0.2):
    """
    Preprocess stock data for machine learning model.
    
    Args:
        df (pd.DataFrame): Input stock data
        look_back (int, optional): Number of previous days to use for prediction. Defaults to 90.
        prediction_days (int, optional): Number of future days to predict. Defaults to 5.
        test_size (float, optional): Proportion of data to use for validation. Defaults to 0.2.
    
    Returns:
        tuple: Processed training and validation data
    """
    # Ensure data is numeric and clean
    numeric_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN
    df.dropna(subset=numeric_columns, inplace=True)
    
    # Create scalers for each column
    scalers = {}
    scaled_data = df.copy()
    for col in numeric_columns:
        # Use RobustScaler to handle outliers better
        scaler = RobustScaler()
        scaled_data[col] = scaler.fit_transform(df[[col]])
        scalers[col] = {
            'scaler': scaler,
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # Create sequences
    X, y = create_sequences(scaled_data, look_back, prediction_days)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=False  # Preserve time series order
    )
    
    return X_train, X_val, y_train, y_val, scalers

def evaluate_predictions(y_val, predictions, scalers, output_days=5):
    """
    Evaluate model predictions with robust error handling and scaling.
    
    Args:
        y_val (np.ndarray): Actual values
        predictions (np.ndarray): Predicted values
        scalers (dict): Dictionary of scalers used for each feature
        output_days (int, optional): Number of days to evaluate. Defaults to 5.
    
    Returns:
        list: Detailed prediction statistics for each day
    """
    try:
        # Ensure y_val and predictions are numpy arrays
        y_val = np.array(y_val)
        predictions = np.array(predictions)
        
        # Reshape if necessary
        if y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Validate shapes
        if y_val.shape != predictions.shape:
            print(f"Shape mismatch: y_val {y_val.shape}, predictions {predictions.shape}")
            y_val = y_val[:predictions.shape[0], :predictions.shape[1]]
        
        # Compute day-wise statistics
        day_stats = []
        for day in range(output_days):
            try:
                # Extract day's actual and predicted values
                actual_day = y_val[:, day]
                predicted_day = predictions[:, day]
                
                # Handle zero or near-zero values
                actual_day = np.where(np.abs(actual_day) < 1e-10, 1e-10, actual_day)
                
                # Compute MAPE with robust error handling
                mape_values = np.abs((actual_day - predicted_day) / actual_day) * 100
                mape_values = np.nan_to_num(mape_values, nan=100, posinf=100, neginf=100)
                mape = np.mean(mape_values)
                
                # Compute mean values
                mean_actual = np.mean(actual_day)
                mean_predicted = np.mean(predicted_day)
                
                day_stats.append({
                    'day': day + 1,
                    'mape': mape,
                    'mean_actual': mean_actual,
                    'mean_predicted': mean_predicted
                })
            
            except Exception as day_error:
                print(f"Error processing day {day + 1}: {day_error}")
                day_stats.append({
                    'day': day + 1,
                    'mape': np.nan,
                    'mean_actual': np.nan,
                    'mean_predicted': np.nan
                })
        
        return day_stats
    
    except Exception as e:
        print(f"Comprehensive error in evaluate_predictions: {e}")
        return [{'day': day + 1, 'mape': np.nan, 'mean_actual': np.nan, 'mean_predicted': np.nan} 
                for day in range(output_days)]

def make_future_predictions(model, last_sequence, scaler, base_price):
    """Make future predictions using percentage changes."""
    future_pct_changes = []
    current_sequence = last_sequence.copy()
    
    for _ in range(PREDICTION_DAYS):
        # Reshape for prediction
        current_reshape = np.expand_dims(current_sequence, axis=0)
        
        # Predict next percentage change
        next_pct_change_pred = model.predict(current_reshape, verbose=0)[0][0]
        future_pct_changes.append(next_pct_change_pred)
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = next_pct_change_pred
    
    # Convert percentage changes back to prices
    future_prices = [base_price]
    for pct_change in future_pct_changes:
        # Inverse transform percentage change
        next_price = future_prices[-1] * (1 + pct_change)
        future_prices.append(next_price)
    
    return future_prices[1:]  # Exclude the base price

def calculate_prediction_uncertainty(predictions):
    """
    Calculate prediction uncertainty using ensemble variance.
    
    Args:
        predictions (np.array): Predictions from multiple models
    
    Returns:
        tuple: Mean prediction, prediction uncertainty
    """
    # Calculate mean prediction
    mean_prediction = np.mean(predictions, axis=0)
    
    # Calculate prediction variance (uncertainty)
    prediction_variance = np.var(predictions, axis=0)
    
    # Calculate confidence interval (e.g., standard error)
    prediction_uncertainty = np.sqrt(prediction_variance)
    
    return mean_prediction, prediction_uncertainty

def advanced_ensemble_prediction(X_val, y_val, models, scalers, ticker, output_days=5):
    """
    Make predictions using an ensemble of models and analyze results.
    
    Args:
        X_val (np.array): Validation input data
        y_val (np.array): Validation target data
        models (list): List of trained models
        scalers (dict): Dictionary of scalers used for each feature
        ticker (str): Stock ticker symbol
        output_days (int, optional): Number of days to predict. Defaults to 5.
    
    Returns:
        tuple: Predictions, actual values, and statistical summary
    """
    try:
        # Ensemble prediction
        predictions_list = []
        for model in models:
            model_predictions = model.predict(X_val)
            predictions_list.append(model_predictions)
        
        # Compute ensemble mean and uncertainty
        predictions = np.mean(predictions_list, axis=0)
        
        # Compute prediction uncertainty
        uncertainty = np.std(predictions_list, axis=0)
        
        # Evaluate predictions
        prediction_stats = evaluate_predictions(y_val, predictions, scalers, output_days)
        
        # Print prediction statistics
        print(f"\n{ticker} Prediction Summary:")
        for stat in prediction_stats:
            print(f"Day {stat['day']}:")
            print(f"  Mean Absolute Percentage Error (MAPE): {stat['mape']:.2f}%")
            print(f"  Mean Actual Price: ${stat['mean_actual']:.2f}")
            print(f"  Mean Predicted Price: ${stat['mean_predicted']:.2f}")
        
        return predictions, y_val, prediction_stats
    
    except Exception as e:
        print(f"Error in advanced_ensemble_prediction: {e}")
        return None, None, None

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
    Download stock data, skipping if the end date is a market holiday.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str or datetime): Start date for data download
        end_date (str or datetime): End date for data download
    
    Returns:
        pd.DataFrame or None: Stock data, or None if download is skipped
    """
    # Convert end_date to pandas Timestamp
    end_date_ts = pd.Timestamp(end_date)
    
    # Check if end date is a market holiday or weekend
    if is_market_holiday(end_date_ts):
        print(f"Skipping data download for {ticker}: {end_date} is a market holiday or weekend")
        return None
    
    try:
        # Existing data download logic
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
        
        # Reset index to make Date a column
        stock_data.reset_index(inplace=True)
        
        return stock_data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

def process_stock_data(stock_data, look_back=90, prediction_days=5):
    """
    Process stock data for machine learning model with improved scaling.
    
    Args:
        stock_data (pd.DataFrame): Input stock price data
        look_back (int): Number of previous days to use for prediction
        prediction_days (int): Number of future days to predict
    
    Returns:
        tuple: Processed features, targets, and scaler
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    
    # Validate input data
    if stock_data is None or stock_data.empty:
        print("Error: No stock data available")
        return None, None, None
    
    # Ensure numeric columns
    numeric_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in numeric_columns:
        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
    
    # Drop rows with NaN
    stock_data.dropna(subset=numeric_columns, inplace=True)
    
    def generate_proxy_market_data(df):
        """
        Generate proxy market data columns if they don't exist.
        This method creates synthetic market indicators based on the stock's own data.
        """
        # Proxy for market data using the stock's own features
        df['Close_SPY'] = df['Close'].rolling(window=20).mean()
        df['High_SPY'] = df['High'].rolling(window=20).max()
        df['Low_SPY'] = df['Low'].rolling(window=20).min()
        
        # Market correlation proxy
        df['Market_Correlation'] = df['Close'].rolling(window=20).corr(df['Volume'])
        
        return df
    
    # Generate proxy market data
    stock_data = generate_proxy_market_data(stock_data)
    
    # Advanced Technical Indicators
    def calculate_advanced_indicators(df):
        # Exponential Moving Averages with multiple windows
        for window in [10, 20, 50, 100, 200]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Relative Strength Index (RSI) with multiple periods
        def compute_rsi(data, periods=[14, 21, 28, 50]):
            for period in periods:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        compute_rsi(df)
        
        # Bollinger Bands with multiple standard deviation multipliers
        for std_mult in [1.5, 2, 2.5]:
            rolling_mean = df['Close'].rolling(window=20).mean()
            rolling_std = df['Close'].rolling(window=20).std()
            df[f'BB_Upper_{std_mult}'] = rolling_mean + (std_mult * rolling_std)
            df[f'BB_Lower_{std_mult}'] = rolling_mean - (std_mult * rolling_std)
        
        # MACD with signal line and histogram
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        # Advanced Momentum Indicators
        for period in [10, 20, 30, 50]:
            df[f'ROC_{period}'] = df['Close'].pct_change(periods=period) * 100
            df[f'Momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Volatility indicators
        df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Volume indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_MA_50'] = df['Volume'].rolling(window=50).mean()
        
        # Percentage changes and log returns
        for col in ['Close', 'Volume']:
            df[f'{col}_Pct_Change'] = df[col].pct_change()
            df[f'{col}_Log_Return'] = np.log(df[col] / df[col].shift(1))
        
        # Trend indicators
        df['Trend_Strength'] = np.abs(df['Close'] - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=50).std()
        
        return df
    
    # Calculate advanced indicators
    stock_data = calculate_advanced_indicators(stock_data)
    
    # Drop rows with NaN after indicator calculation
    stock_data.dropna(inplace=True)
    
    # Select features for model
    feature_columns = [
        # Price and Volume Basics
        'Close', 'Open', 'High', 'Low', 'Volume',
        
        # Proxy Market Data
        'Close_SPY', 'High_SPY', 'Low_SPY',
        
        # Moving Averages
        'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
        
        # RSI Indicators
        'RSI_14', 'RSI_21', 'RSI_28', 'RSI_50',
        
        # Bollinger Bands
        'BB_Upper_1.5', 'BB_Lower_1.5',
        'BB_Upper_2', 'BB_Lower_2',
        
        # MACD Indicators
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        
        # Stochastic Oscillator
        'Stochastic_K', 'Stochastic_D',
        
        # Momentum and Rate of Change
        'ROC_10', 'ROC_20', 'ROC_30', 'ROC_50',
        'Momentum_10', 'Momentum_20', 'Momentum_30', 'Momentum_50',
        
        # Volatility Indicators
        'ATR', 'Volatility',
        
        # Volume Indicators
        'OBV', 'Volume_MA_20', 'Volume_MA_50',
        
        # Percentage Changes and Log Returns
        'Close_Pct_Change', 'Volume_Pct_Change',
        'Close_Log_Return', 'Volume_Log_Return',
        
        # Cross-market and Trend Features
        'Market_Correlation', 'Trend_Strength'
    ]
    
    # Ensure all selected features exist
    available_features = [col for col in feature_columns if col in stock_data.columns]
    
    # Print warning if some features are missing
    missing_features = set(feature_columns) - set(available_features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    # Prepare features
    features = stock_data[available_features]
    
    # Robust scaling
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences for LSTM
    def create_sequences(data, look_back, prediction_days):
        X, y = [], []
        for i in range(len(data) - look_back - prediction_days + 1):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back:i+look_back+prediction_days, 0])  # Use Close price
        return np.array(X), np.array(y)
    
    # Create sequences
    X, y = create_sequences(scaled_features, look_back, prediction_days)
    
    # Diagnostic prints
    print("Features used:", available_features)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    return X, y, scaler

def plot_stock_predictions(actual_values, predictions, ticker, scalers, output_days=5):
    """
    Plot actual stock prices and predicted stock prices.
    
    Args:
        actual_values (np.array): Actual stock price values
        predictions (np.array): Predicted stock price values
        ticker (str): Stock ticker symbol
        scalers (dict): Dictionary of scalers used for each feature
        output_days (int, optional): Number of days to predict. Defaults to 5.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Ensure we have the correct scaler
    if isinstance(scalers, dict):
        close_scaler_info = scalers.get('Close', None)
        if close_scaler_info is None:
            close_scaler_info = next(iter(scalers.values()), None)
        
        # Extract the actual scaler
        close_scaler = close_scaler_info['scaler'] if isinstance(close_scaler_info, dict) else close_scaler_info
    else:
        close_scaler = scalers
    
    # Reshape inputs to ensure compatibility
    actual_values_reshaped = actual_values.reshape(-1, actual_values.shape[-1])
    predictions_reshaped = predictions.reshape(-1, predictions.shape[-1])
    
    # Inverse transform to get original scale
    try:
        # Ensure scale_ and center_ are compatible
        if hasattr(close_scaler, 'scale_') and hasattr(close_scaler, 'center_'):
            scale = np.atleast_1d(close_scaler.scale_)
            center = np.atleast_1d(close_scaler.center_)
            
            # Broadcast to match input shape
            if len(scale) == 1:
                scale = np.repeat(scale, actual_values_reshaped.shape[1])
            if len(center) == 1:
                center = np.repeat(center, actual_values_reshaped.shape[1])
            
            # Inverse transform manually
            actual_original = (actual_values_reshaped * scale) + center
            predictions_original = (predictions_reshaped * scale) + center
            
            # Print detailed scaling information
            print("\nScaling Diagnostic:")
            print(f"scale used: {scale}")
            print(f"center used: {center}")
            print(f"actual_original shape: {actual_original.shape}")
            print(f"predictions_original shape: {predictions_original.shape}")
        else:
            # Fallback to direct comparison if scaling fails
            actual_original = actual_values_reshaped
            predictions_original = predictions_reshaped
            print("Warning: Could not perform scaling. Using raw values.")
    except Exception as e:
        print(f"Error during manual scaling: {e}")
        actual_original = actual_values_reshaped
        predictions_original = predictions_reshaped
    
    # Ensure predictions and actual values have the same shape
    if predictions.shape != actual_values.shape:
        print(f"Warning: Shape mismatch. Predictions: {predictions.shape}, Actual: {actual_values.shape}")
        min_rows = min(predictions.shape[0], actual_values.shape[0])
        predictions = predictions[:min_rows]
        actual_values = actual_values[:min_rows]

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot actual and predicted values for each day
    for day in range(output_days):
        plt.plot(actual_original[:, day], label=f'Actual Day {day+1}', marker='o')
        plt.plot(predictions_original[:, day], label=f'Predicted Day {day+1}', marker='x', linestyle='--')
    
    plt.title(f'{ticker} Stock Price Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'{ticker}_stock_predictions.png')
    plt.close()
    
    # Print some statistical comparisons
    print("\nPrediction vs Actual Statistics:")
    for day in range(output_days):
        # Ensure we have matching shapes
        actual_day = actual_original[:, day]
        predicted_day = predictions_original[:, day]
        
        # Compute MAPE with error handling
        try:
            mape = np.mean(np.abs((actual_day - predicted_day) / actual_day)) * 100
            print(f"Day {day+1}:")
            print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
            print(f"  Mean Actual Price: ${actual_day.mean():.2f}")
            print(f"  Mean Predicted Price: ${predicted_day.mean():.2f}")
        except Exception as e:
            print(f"Day {day+1}: Error computing statistics - {e}")

def predict_stock_prices(tickers, start_date, end_date):
    """
    Predict stock prices for given tickers over a specified date range.
    
    Args:
        tickers (list): List of stock ticker symbols to predict
        start_date (str): Start date for data download
        end_date (str): End date for data download
    
    Returns:
        dict: Prediction results for each ticker
    """
    prediction_results = {}
    
    for ticker in tickers:
        try:
            # Download or load stock data
            stock_data = download_stock_data(ticker, start_date, end_date)
            
            # Add technical indicators
            stock_data = calculate_technical_indicators(stock_data)
            stock_data = calculate_advanced_indicators(stock_data)
            
            # Process stock data
            X, y, scaler = process_stock_data(stock_data)
            
            # Check if data processing was successful
            if X is None or y is None or scaler is None:
                print(f"Skipping {ticker} due to data processing issues")
                continue
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train ensemble models
            models, _ = train_ensemble_model(X_train, y_train, X_val, y_val, scaler, num_models=5, epochs=100, batch_size=32)
            
            # Make predictions
            predictions, actual_values, stats = advanced_ensemble_prediction(X_val, y_val, models, scaler, ticker)
            
            # Plot predictions
            plot_stock_predictions(actual_values, predictions, ticker, scaler)
            
            # Store results
            prediction_results[ticker] = {
                'predictions': predictions,
                'actual_values': actual_values,
                'stats': stats
            }
        
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    return prediction_results

# Update the script's main execution
if __name__ == '__main__':
    # Define tickers to predict
    tickers_to_predict = ['SPY']
    
    # Specify date range
    start_date = '2020-01-02'
    end_date = (pd.Timestamp.today() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Run prediction workflow
    prediction_results = predict_stock_prices(
        tickers=tickers_to_predict, 
        start_date=start_date, 
        end_date=end_date
    )