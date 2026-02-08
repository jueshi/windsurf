import pandas as pd
import os

# Create sample stock data
data = {
    'Date': pd.date_range(start='2025-01-01', periods=10).strftime('%Y-%m-%d'),
    'Open': [100.123456, 101.1, 102.12, 103.123, 104.1234, 105.12345, 106.1, 107.12, 108.123, 109.1234],
    'High': [110.123456, 111.1, 112.12, 113.123, 114.1234, 115.12345, 116.1, 117.12, 118.123, 119.1234],
    'Low': [90.123456, 91.1, 92.12, 93.123, 94.1234, 95.12345, 96.1, 97.12, 98.123, 99.1234],
    'Close': [105.123456, 106.1, 107.12, 108.123, 109.1234, 110.12345, 111.1, 112.12, 113.123, 114.1234],
    'Volume': [1000000, 1100000, 1200000, 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
}

df = pd.DataFrame(data)

# Create output directory if it doesn't exist
os.makedirs('stock_data', exist_ok=True)

# Save the data with different formatting options
print("Creating sample stock data files with different formatting options...")

# 1. Default formatting
df.to_csv('stock_data/SAMPLE_default.tsv', sep='\t', index=False)
print("Created stock_data/SAMPLE_default.tsv")

# 2. With float_format='%.2f'
df.to_csv('stock_data/SAMPLE_float_format.tsv', sep='\t', index=False, float_format='%.2f')
print("Created stock_data/SAMPLE_float_format.tsv")

# 3. With rounding + float_format
df_rounded = df.copy()
numeric_cols = df_rounded.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
for col in numeric_cols:
    df_rounded[col] = df_rounded[col].round(2)
df_rounded.to_csv('stock_data/SAMPLE_rounded_float_format.tsv', sep='\t', index=False, float_format='%.2f')
print("Created stock_data/SAMPLE_rounded_float_format.tsv")

# Print the first few lines of each file to compare
print("\nComparing the first few lines of each file:")

files = [
    'stock_data/SAMPLE_default.tsv',
    'stock_data/SAMPLE_float_format.tsv',
    'stock_data/SAMPLE_rounded_float_format.tsv'
]

for file_path in files:
    print(f"\n{file_path}:")
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:  # Print first 3 lines
                print(f"  {line.strip()}")
            else:
                break

print("\nCheck these files to see which formatting option works best for your needs.")
