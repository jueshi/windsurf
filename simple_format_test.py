import pandas as pd

# Create sample data with different decimal places
data = {
    'Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05'],
    'Open': [100.123456, 101.1, 102.12, 103.123, 104.1234],
    'High': [110.123456, 111.1, 112.12, 113.123, 114.1234],
    'Low': [90.123456, 91.1, 92.12, 93.123, 94.1234],
    'Close': [105.123456, 106.1, 107.12, 108.123, 109.1234],
    'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
}

df = pd.DataFrame(data)
print("Original data:")
print(df)

# Test file path
test_file = 'simple_format_test.tsv'

# Round numeric columns to 2 decimal places
numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
for col in numeric_cols:
    df[col] = df[col].round(2)

# Save with consistent float format
df.to_csv(test_file, sep='\t', index=False, float_format='%.2f')

print(f"\nData saved to {test_file}")

# Read the raw file content to check formatting
print("\nRaw file content:")
with open(test_file, 'r') as f:
    content = f.read()
    print(content)

# Read the file back with pandas
df_read = pd.read_csv(test_file, sep='\t')
print("\nData read back from file:")
print(df_read)

print("\nAll numeric columns should now have exactly 2 decimal places.")
