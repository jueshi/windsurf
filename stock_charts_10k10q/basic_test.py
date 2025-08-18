import os
import sys

# Print current working directory and Python path for debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

try:
    # Import the data manager
    print("Attempting to import StockDataManager...")
    from data_manager import StockDataManager
    print("Successfully imported StockDataManager")
    
    # Create an instance
    print("Creating StockDataManager instance...")
    manager = StockDataManager()
    print("Successfully created StockDataManager instance")
    
    # Print some basic info
    print(f"Data directory: {manager.data_dir}")
    
    # Try to access a method
    print("Testing a method...")
    ticker = "AAPL"
    data_path = manager._get_data_path(ticker)
    print(f"Data path for {ticker}: {data_path}")
    
    print("Test completed successfully")
    
except Exception as e:
    print(f"Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
