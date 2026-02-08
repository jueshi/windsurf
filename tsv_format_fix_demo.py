import os
import pandas as pd
import numpy as np

def demonstrate_tsv_formatting():
    """
    Demonstrate how to fix TSV formatting issues with consistent decimal places
    """
    print("TSV Formatting Fix Demonstration")
    print("=" * 50)
    
    # Create sample data with inconsistent decimal places
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=5).strftime('%Y-%m-%d'),
        'Open': [100.123456, 101.1, 102.12, 103.123, 104.1234],
        'High': [110.123456, 111.1, 112.12, 113.123, 114.1234],
        'Low': [90.123456, 91.1, 92.12, 93.123, 94.1234],
        'Close': [105.123456, 106.1, 107.12, 108.123, 109.1234],
        'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
    }
    
    df = pd.DataFrame(data)
    print("\nOriginal data (with inconsistent decimal places):")
    print(df)
    
    # Save with default formatting (problem)
    problem_file = 'problem_format.tsv'
    df.to_csv(problem_file, sep='\t', index=False)
    
    # Save with only float_format (partial solution)
    partial_fix_file = 'partial_fix_format.tsv'
    df.to_csv(partial_fix_file, sep='\t', index=False, float_format='%.2f')
    
    # Save with rounding + float_format (complete solution)
    complete_fix_file = 'complete_fix_format.tsv'
    df_rounded = df.copy()
    numeric_cols = df_rounded.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_cols:
        df_rounded[col] = df_rounded[col].round(2)
    df_rounded.to_csv(complete_fix_file, sep='\t', index=False, float_format='%.2f')
    
    # Compare the raw file contents
    print("\nComparing raw file contents:")
    
    print("\n1. PROBLEM FILE (default formatting):")
    with open(problem_file, 'r') as f:
        problem_content = f.read()
    print(problem_content)
    
    print("\n2. PARTIAL FIX (float_format only):")
    with open(partial_fix_file, 'r') as f:
        partial_fix_content = f.read()
    print(partial_fix_content)
    
    print("\n3. COMPLETE FIX (rounding + float_format):")
    with open(complete_fix_file, 'r') as f:
        complete_fix_content = f.read()
    print(complete_fix_content)
    
    print("\nConclusion:")
    print("The complete fix (rounding + float_format) ensures all numeric values")
    print("have exactly 2 decimal places, creating consistent column alignment")
    print("in the TSV file.")

if __name__ == "__main__":
    demonstrate_tsv_formatting()
