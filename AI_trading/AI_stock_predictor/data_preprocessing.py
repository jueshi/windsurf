import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as sk_train_test_split

def clean_stock_data(stock_data):
    """
    Clean and preprocess stock data
    
    Args:
        stock_data (pd.DataFrame): Raw stock price data
    
    Returns:
        pd.DataFrame: Cleaned stock data
    """
    # Drop rows with missing values
    stock_data.dropna(inplace=True)
    
    # Select relevant columns
    columns_to_use = ['Close', 'Volume', 'Open', 'High', 'Low']
    
    # Ensure all required columns exist
    for col in columns_to_use:
        if col not in stock_data.columns:
            raise ValueError(f"Required column '{col}' not found in stock data")
    
    return stock_data[columns_to_use]

def create_sliding_windows(data, window_size=60, forecast_horizon=30):
    """
    Create sliding windows for time series prediction
    
    Args:
        data (pd.DataFrame): Cleaned stock data
        window_size (int): Number of historical time steps to use for prediction
        forecast_horizon (int): Number of future time steps to predict
    
    Returns:
        tuple: X (input sequences), y (target values)
    """
    # Convert DataFrame to numpy array
    data_array = data.values
    
    # Normalize each feature independently
    scalers = [MinMaxScaler() for _ in range(data_array.shape[1])]
    normalized_data = np.zeros_like(data_array, dtype=float)
    
    for i in range(data_array.shape[1]):
        normalized_data[:, i] = scalers[i].fit_transform(data_array[:, i].reshape(-1, 1)).flatten()
    
    # Create sliding windows
    X, y = [], []
    for i in range(len(normalized_data) - window_size - forecast_horizon + 1):
        # Input sequence: last 'window_size' time steps
        X.append(normalized_data[i:i+window_size])
        
        # Target: mean of next 'forecast_horizon' time steps for the first column (Close price)
        y.append(np.mean(normalized_data[i+window_size:i+window_size+forecast_horizon, 0]))
    
    return np.array(X), np.array(y)

def normalize_data(X, scaler=None):
    """Normalize input data"""
    # Flatten the first two dimensions
    original_shape = X.shape
    X_2d = X.reshape(-1, original_shape[-1])
    
    if scaler is None:
        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X_2d)
    else:
        X_normalized = scaler.transform(X_2d)
    
    # Reshape back to original dimensions
    X_normalized = X_normalized.reshape(original_shape)
    
    return X_normalized, scaler

def train_test_split(X, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    return sk_train_test_split(X, test_size=test_size, random_state=random_state)

def train_test_split_custom(data, test_size=0.2, shuffle=False):
    """
    Split data into training and testing sets
    
    Args:
        data (np.ndarray): Input data to split
        test_size (float): Proportion of data to use for testing
        shuffle (bool): Whether to shuffle data before splitting
    
    Returns:
        tuple: Training and testing data
    """
    # Determine split index
    split_idx = int(len(data) * (1 - test_size))
    
    # Shuffle if specified
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(data)
    
    # Split data
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data
