import pandas as pd
import os
import sys
import traceback

def csv_add_Jtol_freq_spec(csv_file):
    """
    Add Jtol_freq and spec columns to a CSV file based on JTOL_MAX patterns.
    
    Args:
        csv_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Modified DataFrame with new columns
    """
    # Clean up the file path - remove any quotes and normalize
    csv_file = os.path.normpath(csv_file.strip('" \''))
    
    print(f"Attempting to read file: {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Input file not found: {csv_file}")
        
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read CSV with {len(df)} rows")
        print(f"Columns found: {', '.join(df.columns)}")
        
        # Print first few rows of Result_Name and Result_Value columns
        print("\nFirst few rows of relevant columns:")
        if 'Result_Name' in df.columns and 'Result_Value' in df.columns:
            print(df[['Result_Name', 'Result_Value']].head())
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        raise
    
    try:
        # Look for JTOL_MAX in Result_Name column
        if 'Result_Name' not in df.columns:
            raise ValueError("Required column 'Result_Name' not found in CSV")
            
        # Find rows with JTOL_MAX in Result_Name
        jtol_mask = df['Result_Name'].str.contains('JTOL_MAX_', case=False, na=False)
        if not jtol_mask.any():
            raise ValueError("No 'JTOL_MAX_' values found in Result_Name column")
            
        print(f"\nFound {jtol_mask.sum()} rows with JTOL_MAX_ pattern")
        print("\nUnique JTOL_MAX values found:")
        print(df.loc[jtol_mask, 'Result_Name'].unique())
        
        # Create new DataFrame with Jtol_freq and spec columns
        new_cols = pd.DataFrame(index=df.index)
        
        # Add Jtol_freq column
        new_cols['Jtol_freq'] = ''
        new_cols.loc[jtol_mask, 'Jtol_freq'] = df.loc[jtol_mask, 'Result_Name'].str.replace('JTOL_MAX_', '', case=False)
        
        # Add spec column with default NaN values
        new_cols['spec'] = float('nan')
        
        # Set spec values based on JTOL_MAX patterns
        new_cols.loc[df['Result_Name'].str.contains('JTOL_MAX_12|JTOL_MAX_4|JTOL_MAX_40', case=False, na=False), 'spec'] = 0.05
        new_cols.loc[df['Result_Name'].str.contains('JTOL_MAX_1.33', case=False, na=False), 'spec'] = 0.15
        new_cols.loc[df['Result_Name'].str.contains('JTOL_MAX_0.04', case=False, na=False), 'spec'] = 5.0
        
        # Combine new columns with original DataFrame
        df = pd.concat([new_cols, df], axis=1)
        
        print("\nSummary of spec values assigned:")
        print(df.loc[jtol_mask, ['Result_Name', 'Jtol_freq', 'spec']].drop_duplicates())
        
        return df
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

def process_csv_file(input_csv, output_csv=None):
    """
    Process a CSV file by adding Jtol_freq and spec columns.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_csv (str, optional): Path to save the output CSV file. If None, will overwrite input file.
    """
    try:
        print(f"\nProcessing input file: {input_csv}")
        df = csv_add_Jtol_freq_spec(input_csv)
        output_path = output_csv if output_csv else input_csv
        # Clean up output path
        output_path = os.path.normpath(str(output_path).strip('" \''))
        print(f"Saving to output file: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"Successfully saved processed CSV to: {output_path}")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else None
        process_csv_file(input_csv, output_csv)
    else:
        print("Usage: python csv_add_Jtol-freq_spec.py input.csv [output.csv]")
        input_csv = r"C:\Users\JueShi\Astera Labs, Inc\Silicon Engineering - 100G Rx KR Char Data for AWS Report\Test1\ppm data\Kr_Rx_test1_TT102_ppm_sweep_NVNT_0x19e001b9062040000001212_PMD3_Lane1_100G.csv"
        output_csv = r"C:\Users\JueShi\Astera Labs, Inc\Silicon Engineering - 100G Rx KR Char Data for AWS Report\Test1\ppm data\Kr_Rx_test1_TT102_ppm_sweep_NVNT_0x19e001b9062040000001212_PMD3_Lane1_100G_add_spec.csv"
        process_csv_file(input_csv, output_csv)