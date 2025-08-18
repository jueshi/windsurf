import os
import pandas as pd
import numpy as np
import sys

# Add the stock_charts_10k10q directory to the path
sys.path.append('stock_charts_10k10q')

# Import the StockDataManager class
from stock_charts_10k10q.data_manager import StockDataManager

def test_formatting():
    """Test the data formatting fix by creating and saving sample data"""
    # Create sample data with different decimal places
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=10),
        'Open': [100.123456, 101.1, 102.12, 103.123, 104.1234, 105.12345, 106.1, 107.12, 108.123, 109.1234],
        'High': [110.123456, 111.1, 112.12, 113.123, 114.1234, 115.12345, 116.1, 117.12, 118.123, 119.1234],
        'Low': [90.123456, 91.1, 92.12, 93.123, 94.1234, 95.12345, 96.1, 97.12, 98.123, 99.1234],
        'Close': [105.123456, 106.1, 107.12, 108.123, 109.1234, 110.12345, 111.1, 112.12, 113.123, 114.1234],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
    }
    
    df = pd.DataFrame(data)
    
    # Create a StockDataManager instance
    manager = StockDataManager()
    
    # Test file path
    test_file = os.path.join('stock_data', 'TEST_FORMAT_data.tsv')
    
    # Save the data using our new method
    manager._save_data_with_consistent_format(df, test_file)
    
    print(f"Data saved to {test_file}")
    
    # Read the file and print the first few lines to check formatting
    print("\nFile content (first few lines):")
    try:
        # Read the file using pandas to ensure proper display
        test_data = pd.read_csv(test_file, sep='\t')
        print("\nColumn names:")
        print(test_data.columns.tolist())
        
        print("\nFirst 5 rows:")
        print(test_data.head(5).to_string(index=False))
        
        # Check for consistent decimal places
        numeric_cols = test_data.select_dtypes(include=['float64', 'float32']).columns
        print("\nChecking decimal place consistency:")
        for col in numeric_cols:
            # Extract decimal parts and their lengths
            decimal_parts = test_data[col].astype(str).str.split('.').str[1]
            unique_lengths = decimal_parts.str.len().unique()
            print(f"{col}: decimal places lengths = {unique_lengths}")
    except Exception as e:
        print(f"Error reading the test file: {e}")
        
        # Fallback to reading raw lines
        with open(test_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()[:10]]
        
        print("\nRaw file content (first few lines):")
        for line in lines:
            print(line)

if __name__ == "__main__":
    test_formatting()
