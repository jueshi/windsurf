import os
import pandas as pd
import glob

def fix_stock_data_formatting():
    """Fix the formatting of all stock data files to ensure consistent decimal places"""
    # Define the stock data directory
    stock_data_dir = 'stock_data'
    
    # Find all TSV files in the stock data directory
    tsv_files = glob.glob(os.path.join(stock_data_dir, '*_stock_data.tsv'))
    
    if not tsv_files:
        print(f"No stock data files found in {stock_data_dir}")
        return
    
    print(f"Found {len(tsv_files)} stock data files to process")
    
    for file_path in tsv_files:
        ticker = os.path.basename(file_path).split('_')[0]
        print(f"\nProcessing {ticker} data file: {file_path}")
        
        try:
            # Read the data
            df = pd.read_csv(file_path, sep='\t')
            
            print(f"  Original data shape: {df.shape}")
            
            # Round numeric columns to 2 decimal places
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            for col in numeric_cols:
                df[col] = df[col].round(2)
            
            # Save with consistent float format
            df.to_csv(file_path, sep='\t', index=False, float_format='%.2f')
            
            print(f"  Fixed formatting in {file_path}")
            
            # Verify the fix by reading a few lines directly from the file
            with open(file_path, 'r') as f:
                lines = [next(f) for _ in range(min(3, len(df) + 1))]
            
            print("  Sample lines from fixed file:")
            for i, line in enumerate(lines):
                print(f"    Line {i+1}: {line.strip()}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print("\nAll stock data files have been processed")

if __name__ == "__main__":
    fix_stock_data_formatting()
