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
        data (pd.DataFrame): Input stock price data
        window_size (int): Number of time steps to use for prediction
        forecast_horizon (int): Number of future time steps to predict
    
    Returns:
        tuple: Preprocessed input features (X), target values (y), scalers, validation indices, and original data
    """
    # Convert to numpy array for easier manipulation
    data_array = data.to_numpy()
    
    # Print comprehensive date range information
    print("\n===== Data Range and Sliding Window Analysis =====")
    print("Full Dataset:")
    print(f"  Total Data Points: {len(data)}")
    print(f"  Date Range: {data.index[0]} to {data.index[-1]}")
    print(f"  Total Duration: {(data.index[-1] - data.index[0]).days} days")
    
    # Calculate train/validation split
    train_split = int(len(data_array) * 0.8)
    
    # Detailed split information
    print("\nData Split Details:")
    print("  Training Period:")
    print(f"    Start: {data.index[0]}")
    print(f"    End: {data.index[train_split-1]}")
    print(f"    Duration: {(data.index[train_split-1] - data.index[0]).days} days")
    print(f"    Data Points: {train_split}")
    
    print("  Validation Period:")
    print(f"    Start: {data.index[train_split]}")
    print(f"    End: {data.index[-1]}")
    print(f"    Duration: {(data.index[-1] - data.index[train_split]).days} days")
    print(f"    Data Points: {len(data) - train_split}")
    
    # Sliding Window Specifics
    print("\nSliding Window Configuration:")
    print(f"  Window Size: {window_size} days")
    print(f"  Forecast Horizon: {forecast_horizon} days")
    
    # Initialize scalers for each column
    scalers = []
    normalized_data = np.zeros_like(data_array, dtype=float)
    
    # Custom scaling to preserve price relationships
    for i in range(data_array.shape[1]):
        # Skip scaling for binary or categorical columns
        if len(np.unique(data_array[:, i])) <= 2:
            normalized_data[:, i] = data_array[:, i]
            scalers.append(None)
            continue
        
        # Use MinMaxScaler for continuous columns
        scaler = MinMaxScaler()
        
        # Fit and transform, handling potential edge cases
        try:
            normalized_data[:, i] = scaler.fit_transform(data_array[:, i].reshape(-1, 1)).flatten()
        except ValueError:
            # Fallback to manual normalization if MinMaxScaler fails
            min_val = np.min(data_array[:, i])
            max_val = np.max(data_array[:, i])
            normalized_data[:, i] = (data_array[:, i] - min_val) / (max_val - min_val)
            scaler = None
        
        scalers.append(scaler)
    
    # Create sliding windows
    X, y, validation_indices, validation_prices, validation_dates = [], [], [], [], []
    
    # Iterate through the validation period to create sliding windows
    for i in range(train_split, len(normalized_data) - window_size - forecast_horizon + 1):
        # Extract input window
        X_window = normalized_data[i-window_size:i]
        
        # Extract target (next day's closing price)
        y_window = normalized_data[i+window_size, 0]  # Assuming first column is Close price
        
        X.append(X_window)
        y.append(y_window)
        
        # Track validation indices, prices, and dates
        validation_indices.append(i)
        validation_prices.append(data_array[i+window_size, 0])  # Original close price
        validation_dates.append(data.index[i+window_size])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Ensure validation data matches X_test shape
    X_train, X_test = X, X
    y_train, y_test = y, y
    
    # Adjust validation prices and dates
    validation_prices = np.array(validation_prices)
    validation_dates = np.array(validation_dates)
    
    # Detailed Sliding Window Visualization
    print("\nSliding Window Validation Details:")
    print("  Validation Window Information:")
    print(f"    First Validation Window:")
    print(f"      Input Window: {data.index[validation_indices[0]]} to {data.index[validation_indices[0]+window_size-1]}")
    print(f"      Prediction Date: {validation_dates[0]}")
    print(f"      Prediction Price: {validation_prices[0]:.2f}")
    
    print(f"    Last Validation Window:")
    print(f"      Input Window: {data.index[validation_indices[-1]]} to {data.index[validation_indices[-1]+window_size-1]}")
    print(f"      Prediction Date: {validation_dates[-1]}")
    print(f"      Prediction Price: {validation_prices[-1]:.2f}")
    
    print("\n  Validation Data Characteristics:")
    print(f"    Total Validation Windows: {len(validation_indices)}")
    print(f"    Validation Date Range: {validation_dates[0]} to {validation_dates[-1]}")
    print(f"    Price Range: {np.min(validation_prices):.2f} - {np.max(validation_prices):.2f}")
    
    print("\n===== End of Sliding Window Analysis =====")
    
    return X_train, X_test, y_train, y_test, scalers, validation_indices, validation_prices, data_array

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
