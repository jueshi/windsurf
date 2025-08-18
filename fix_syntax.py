# Read the file
with open('c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts.py', 'r') as f:
    content = f.read()

# Fix the syntax error by replacing escaped single quotes
content = content.replace("\\'active_tab\\'", "'active_tab'")

# Write the updated content back to the file
with open('c:\\Users\\juesh\\OneDrive\\Documents\\windsurf\\stock_charts.py', 'w') as f:
    f.write(content)

print("Syntax error fixed successfully.")
