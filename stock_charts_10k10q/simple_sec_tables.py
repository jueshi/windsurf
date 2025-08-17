from sec_api import QueryApi
import pandas as pd
import requests
import sys
import traceback
from dotenv import load_dotenv
import os

def main():
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("SEC_API_KEY")
    
    if not api_key:
        print("Error: SEC_API_KEY not found in environment variables")
        return 1
    
    # Initialize API client
    query_api = QueryApi(api_key=api_key)
    
    # Get ticker from command line or use default
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    print(f"Searching for {ticker} 10-K filings...")
    
    try:
        # Query for the most recent 10-K filing
        query = {
            "query": {"query_string": {"query": f"ticker:{ticker} AND formType:\"10-K\""}},
            "from": "0",
            "size": "1",
            "sort": [{"filedAt": "desc"}]
        }
        
        filings = query_api.get_filings(query)
        
        if not filings or 'filings' not in filings or not filings['filings']:
            print(f"No 10-K filings found for {ticker}")
            return 1
        
        # Get filing details
        filing = filings['filings'][0]
        filing_url = filing['linkToFilingDetails']
        accession_no = filing['accessionNo']
        cik = filing['cik']
        filing_date = filing['filedAt']
        
        print(f"Found filing: {filing_url}")
        print(f"Company CIK: {cik}")
        print(f"Filing Date: {filing_date}")
        
        # Download the HTML content
        print("Downloading filing HTML content...")
        response = requests.get(filing_url)
        
        if response.status_code != 200:
            print(f"Failed to download filing. Status code: {response.status_code}")
            return 1
        
        html_content = response.text
        print(f"Downloaded {len(html_content)} bytes of HTML content")
        
        # Save HTML content for debugging
        with open(f"{ticker}_10k.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Saved HTML content to {ticker}_10k.html")
        
        # Extract tables from HTML
        print("Extracting tables from HTML...")
        try:
            tables = pd.read_html(html_content)
            print(f"Found {len(tables)} tables in the filing")
            
            # Save the first 5 tables to Excel for inspection
            if tables:
                with pd.ExcelWriter(f"{ticker}_tables.xlsx") as writer:
                    for i, table in enumerate(tables[:10]):
                        sheet_name = f"Table_{i}"
                        table.to_excel(writer, sheet_name=sheet_name)
                print(f"Saved first 10 tables to {ticker}_tables.xlsx")
            
            return 0
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            traceback.print_exc()
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
