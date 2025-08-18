import os
import pandas as pd
import sys

def fix_stock_data_format(ticker):
    """Fix the formatting of a stock data file to ensure consistent decimal places"""
    # Define the file path
    file_path = os.path.join('stock_data', f'{ticker}_stock_data.tsv')
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return
    
    print(f"Processing {file_path}...")
    
    try:
        # Read the data
        df = pd.read_csv(file_path, sep='\t')
        
        print(f"Original data shape: {df.shape}")
        print("First few rows of original data:")
        print(df.head(3))
        
        # Round numeric columns to 2 decimal places
        numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        for col in numeric_cols:
            df[col] = df[col].round(2)
        
        # Save with consistent float format
        df.to_csv(file_path, sep='\t', index=False, float_format='%.2f')
        
        print(f"\nFixed formatting in {file_path}")
        
        # Read the file back to verify
        df_fixed = pd.read_csv(file_path, sep='\t')
        print("First few rows after fixing:")
        print(df_fixed.head(3))
        
        # Check for consistent decimal places
        print("\nChecking decimal place consistency:")
        for col in df_fixed.select_dtypes(include=['float64', 'float32']).columns:
            # Convert to string and check decimal places
            str_values = df_fixed[col].astype(str)
            decimal_parts = [val.split('.')[1] if '.' in val else '' for val in str_values[:5]]
            print(f"{col}: sample decimal parts = {decimal_parts}")
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Get ticker from command line argument or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "GOOG"
    fix_stock_data_format(ticker)
