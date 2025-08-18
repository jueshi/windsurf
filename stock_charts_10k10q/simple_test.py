import os
import sys
import pandas as pd
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the data manager
from data_manager import StockDataManager

def test_stock_data_manager():
    """Simple test for StockDataManager functionality"""
    print("Starting StockDataManager test...")
    
    # Initialize the manager
    manager = StockDataManager()
    print("StockDataManager initialized successfully")
    
    # Test loading data for a ticker that should already exist
    ticker = "AAPL"
    print(f"Testing data loading for {ticker}...")
    
    try:
        # Try to load existing data
        data = manager.load_data(ticker)
        if data is not None and not data.empty:
            print(f"Successfully loaded data for {ticker}")
            print(f"Data shape: {data.shape}")
            print(f"Data columns: {data.columns.tolist()}")
            print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        else:
            print(f"No existing data found for {ticker}, will try to download")
            
            # Try downloading data
            data = manager.update_data(ticker, force_download=False)
            if data is not None and not data.empty:
                print(f"Successfully downloaded data for {ticker}")
                print(f"Data shape: {data.shape}")
                print(f"Data columns: {data.columns.tolist()}")
                print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
            else:
                print(f"Failed to download data for {ticker}")
    
    except Exception as e:
        print(f"Error during test: {str(e)}")
    
    print("Test completed")

if __name__ == "__main__":
    test_stock_data_manager()
