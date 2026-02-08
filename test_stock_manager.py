import os
import sys
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the stock_charts_10k10q directory to the path
sys.path.insert(0, os.path.abspath('stock_charts_10k10q'))

# Import the StockDataManager class
from data_manager import StockDataManager

def test_save_data_formatting():
    """Test the _save_data_with_consistent_format method directly"""
    # Create a StockDataManager instance
    manager = StockDataManager()
    
    # Create sample data with different decimal places
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=5),
        'Open': [100.123456, 101.1, 102.12, 103.123, 104.1234],
        'High': [110.123456, 111.1, 112.12, 113.123, 114.1234],
        'Low': [90.123456, 91.1, 92.12, 93.123, 94.1234],
        'Close': [105.123456, 106.1, 107.12, 108.123, 109.1234],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }
    
    df = pd.DataFrame(data)
    print("Original data:")
    print(df.to_string())
    
    # Test file path
    test_file = 'test_stock_manager_format.tsv'
    
    # Save the data using our new method
    manager._save_data_with_consistent_format(df, test_file)
    print(f"\nData saved to {test_file}")
    
    # Read the raw file content to check formatting
    print("\nRaw file content:")
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            print(f"Line {i+1}: {line.strip()}")
    
    # Read the file back with pandas
    df_read = pd.read_csv(test_file, sep='\t')
    print("\nData read back from file:")
    print(df_read.to_string())
    
    # Check for consistent decimal places
    print("\nChecking decimal place consistency:")
    for col in df_read.select_dtypes(include=['float64', 'float32']).columns:
        # Convert to string and check decimal places
        str_values = df_read[col].astype(str)
        decimal_lengths = [len(val.split('.')[1]) if '.' in val else 0 for val in str_values]
        unique_lengths = set(decimal_lengths)
        print(f"{col}: decimal places = {unique_lengths}")

if __name__ == "__main__":
    test_save_data_formatting()
