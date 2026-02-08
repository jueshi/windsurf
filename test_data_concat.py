import pandas as pd
import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def test_data_concatenation():
    """
    Test that data concatenation maintains consistent formatting
    """
    print("Testing data concatenation with consistent formatting")
    print("=" * 60)
    
    # Create sample existing data with inconsistent decimal places
    existing_data = pd.DataFrame({
        'Date': ['2025-01-01', '2025-01-02', '2025-01-03'],
        'Open': [100.123456, 101.1, 102.12],
        'High': [110.123456, 111.1, 112.12],
        'Low': [90.123456, 91.1, 92.12],
        'Close': [105.123456, 106.1, 107.12],
        'Volume': [1000000, 1100000, 1200000]
    })
    
    # Create sample new data with different decimal places
    new_data = pd.DataFrame({
        'Date': ['2025-01-04', '2025-01-05'],
        'Open': [103.5, 104.75],
        'High': [113.25, 114.875],
        'Low': [93.625, 94.375],
        'Close': [108.5, 109.25],
        'Volume': [1300000, 1400000]
    })
    
    print("\nOriginal existing data:")
    print(existing_data)
    
    print("\nOriginal new data:")
    print(new_data)
    
    # Apply the fix: round both datasets before concatenation
    print("\nApplying fix: rounding both datasets before concatenation...")
    
    # Round existing data
    numeric_cols = existing_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_cols:
        existing_data[col] = existing_data[col].round(2)
    
    # Round new data
    numeric_cols = new_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_cols:
        new_data[col] = new_data[col].round(2)
    
    # Concatenate the data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    print("\nCombined data after rounding both datasets:")
    print(combined_data)
    
    # Save to TSV file with consistent float format
    output_file = 'test_concat_output.tsv'
    combined_data.to_csv(output_file, sep='\t', index=False, float_format='%.2f')
    
    # Read the file back to verify formatting
    print(f"\nReading back the saved TSV file ({output_file}):")
    with open(output_file, 'r') as f:
        file_content = f.read()
    print(file_content)
    
    # Check if all numeric values have exactly 2 decimal places
    df_read = pd.read_csv(output_file, sep='\t')
    
    print("\nVerifying decimal places in the saved file:")
    for col in df_read.select_dtypes(include=['float64']).columns:
        # Convert to string and check decimal places
        decimal_places = df_read[col].astype(str).str.split('.').str[1].str.len()
        all_two_places = (decimal_places == 2).all()
        print(f"Column {col}: All values have exactly 2 decimal places? {all_two_places}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_data_concatenation()
