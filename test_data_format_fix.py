import os
import pandas as pd
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Add the stock_charts_10k10q directory to the path
sys.path.insert(0, os.path.abspath('stock_charts_10k10q'))

# Try to import the StockDataManager class
try:
    from data_manager import StockDataManager
    logging.info("Successfully imported StockDataManager")
except ImportError as e:
    logging.error(f"Error importing StockDataManager: {e}")
    sys.exit(1)

def verify_data_format_fix(ticker="GOOG"):
    """
    Verify that the data format fix works correctly for actual stock data files
    """
    print(f"Verifying data format fix for {ticker}")
    print("=" * 60)
    
    # Create a StockDataManager instance
    manager = StockDataManager()
    
    # Get the data file path
    data_path = manager._get_data_path(ticker)
    
    if not os.path.exists(data_path):
        print(f"Data file for {ticker} not found at {data_path}")
        return
    
    print(f"Found data file: {data_path}")
    
    # Read the original data
    try:
        original_data = pd.read_csv(data_path, sep='\t')
        print(f"Original data shape: {original_data.shape}")
        
        # Check the decimal places in the original data
        print("\nChecking decimal places in original data:")
        for col in original_data.select_dtypes(include=['float64']).columns:
            # Convert to string and check decimal places
            decimal_places = original_data[col].astype(str).str.split('.').str[1].str.len().value_counts()
            print(f"Column {col}: Decimal place counts: {decimal_places.to_dict()}")
        
        # Apply the fix: round numeric columns and save with consistent format
        print("\nApplying fix to the data...")
        
        # Make a copy of the data
        fixed_data = original_data.copy()
        
        # Round numeric columns to 2 decimal places
        numeric_cols = fixed_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_cols:
            fixed_data[col] = fixed_data[col].round(2)
        
        # Save with consistent float format to a temporary file
        temp_file = f"{ticker}_fixed_format.tsv"
        fixed_data.to_csv(temp_file, sep='\t', index=False, float_format='%.2f')
        print(f"Saved fixed data to {temp_file}")
        
        # Read the fixed data
        fixed_data_read = pd.read_csv(temp_file, sep='\t')
        
        # Check the decimal places in the fixed data
        print("\nChecking decimal places in fixed data:")
        for col in fixed_data_read.select_dtypes(include=['float64']).columns:
            # Convert to string and check decimal places
            decimal_places = fixed_data_read[col].astype(str).str.split('.').str[1].str.len().value_counts()
            print(f"Column {col}: Decimal place counts: {decimal_places.to_dict()}")
        
        # Compare raw file contents
        print("\nComparing raw file contents:")
        
        print("\n1. Original file (first 5 lines):")
        with open(data_path, 'r') as f:
            lines = [next(f) for _ in range(min(5, len(original_data) + 1))]
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: {line.strip()}")
        
        print("\n2. Fixed file (first 5 lines):")
        with open(temp_file, 'r') as f:
            lines = [next(f) for _ in range(min(5, len(fixed_data_read) + 1))]
        for i, line in enumerate(lines):
            print(f"  Line {i+1}: {line.strip()}")
        
        print("\nVerification completed!")
        
    except Exception as e:
        print(f"Error during verification: {e}")

if __name__ == "__main__":
    # Get ticker from command line arguments or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOG"
    verify_data_format_fix(ticker)
