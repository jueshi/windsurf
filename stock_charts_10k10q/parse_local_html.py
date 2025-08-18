import pandas as pd
import sys
import os
import traceback
from bs4 import BeautifulSoup

def extract_tables_from_html(html_file):
    """Extract tables from a local HTML file"""
    try:
        print(f"Reading HTML file: {html_file}")
        
        # Check if file exists
        if not os.path.exists(html_file):
            print(f"Error: File {html_file} does not exist")
            return None
            
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            
        print(f"Read {len(html_content)} bytes of HTML content")
        
        # Extract tables using pandas
        print("Extracting tables using pandas...")
        try:
            tables = pd.read_html(html_content)
            print(f"Found {len(tables)} tables in the HTML")
            return tables
        except Exception as e:
            print(f"Error extracting tables with pandas: {e}")
            traceback.print_exc()
            
            # Try using BeautifulSoup as a fallback
            print("Trying BeautifulSoup as fallback...")
            soup = BeautifulSoup(html_content, 'html.parser')
            table_tags = soup.find_all('table')
            print(f"Found {len(table_tags)} table tags with BeautifulSoup")
            
            tables = []
            for i, table in enumerate(table_tags):
                try:
                    # Convert table to pandas DataFrame
                    rows = []
                    for tr in table.find_all('tr'):
                        row = []
                        for td in tr.find_all(['td', 'th']):
                            row.append(td.get_text(strip=True))
                        if row:  # Only add non-empty rows
                            rows.append(row)
                            
                    if rows:
                        df = pd.DataFrame(rows)
                        # Use first row as header if it looks like a header
                        if len(df) > 1:
                            df.columns = df.iloc[0]
                            df = df[1:]
                        tables.append(df)
                        print(f"Successfully parsed table {i+1}")
                except Exception as inner_e:
                    print(f"Error parsing table {i+1}: {inner_e}")
                    
            return tables
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def main():
    # Get HTML file path from command line or use default
    html_file = "sample.html"  # Default file
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
    
    # Extract tables from the HTML file
    tables = extract_tables_from_html(html_file)
    if not tables:
        print("Could not extract any tables from the HTML file")
        return 1
    
    # Save the tables to Excel
    output_file = f"{os.path.splitext(os.path.basename(html_file))[0]}_tables.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        # Save up to 30 tables
        for i, table in enumerate(tables[:30]):
            sheet_name = f"Table_{i}"
            table.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Saved {min(len(tables), 30)} tables to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
