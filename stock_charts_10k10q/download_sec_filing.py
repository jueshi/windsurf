import requests
import pandas as pd
import sys
import os

def main():
    # Direct URL to an Apple 10-K filing
    filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    
    print(f"Downloading filing from: {filing_url}")
    
    # Set headers to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # Download the filing
        response = requests.get(filing_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Failed to download filing. Status code: {response.status_code}")
            return 1
        
        html_content = response.text
        print(f"Downloaded {len(html_content)} bytes of HTML content")
        
        # Save HTML content
        with open("apple_10k.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Saved HTML content to apple_10k.html")
        
        # Extract tables from HTML
        print("Extracting tables from HTML...")
        try:
            tables = pd.read_html(html_content)
            print(f"Found {len(tables)} tables in the filing")
            
            # Save the tables to Excel
            if tables:
                with pd.ExcelWriter("apple_tables.xlsx") as writer:
                    # Save up to 30 tables
                    for i, table in enumerate(tables[:30]):
                        sheet_name = f"Table_{i}"
                        table.to_excel(writer, sheet_name=sheet_name)
                print("Saved tables to apple_tables.xlsx")
            
            return 0
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
