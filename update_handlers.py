import re

# Read the file
with open('c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts.py', 'r') as f:
    content = f.read()

# Define the pattern and replacement
pattern = r'if hasattr\(self, \'active_tab\'\) and self\.active_tab == "comparison" and len\(selected_tickers\) > 1:'
replacement = r'if hasattr(self, \'active_tab\') and self.active_tab == "comparison":'

# Replace all occurrences
content = re.sub(pattern, replacement, content)

# Write the updated content back to the file
with open('c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts.py', 'w') as f:
    f.write(content)

print("File updated successfully.")
