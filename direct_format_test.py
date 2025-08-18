import os
import pandas as pd
import numpy as np

def test_consistent_formatting():
    """Test the consistent formatting of data in TSV files"""
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
    
    # Test file path
    test_file = 'test_format_data.tsv'
    
    # Make a copy to avoid modifying the original data
    data_to_save = df.copy()
    
    # Round numeric columns to 2 decimal places
    numeric_cols = data_to_save.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_cols:
        data_to_save[col] = data_to_save[col].round(2)
    
    # Save with consistent float format
    data_to_save.to_csv(test_file, sep='\t', index=False, float_format='%.2f')
    
    print(f"Data saved to {test_file}")
    
    # Read the raw file content
    with open(test_file, 'r') as f:
        content = f.read()
    
    print("\nRaw file content:")
    print(content)
    
    # Read the file back with pandas
    df_read = pd.read_csv(test_file, sep='\t')
    
    # Check for consistent decimal places
    print("\nChecking decimal place consistency:")
    for col in df_read.select_dtypes(include=['float64', 'float32']).columns:
        # Convert to string and check decimal places
        str_values = df_read[col].astype(str)
        decimal_lengths = [len(val.split('.')[1]) if '.' in val else 0 for val in str_values]
        unique_lengths = set(decimal_lengths)
        print(f"{col}: decimal places = {unique_lengths}")
        
        # Print a few sample values
        print(f"Sample values for {col}: {str_values[:5].tolist()}")

if __name__ == "__main__":
    test_consistent_formatting()
