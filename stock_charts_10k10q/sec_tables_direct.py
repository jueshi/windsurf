import requests
import pandas as pd
import sys
import traceback
import re
import os
from bs4 import BeautifulSoup

def get_sec_filing_url(ticker, form_type="10-K"):
    """
    Get the URL for the latest SEC filing of the specified type for a company
    
    Args:
        ticker (str): The stock ticker symbol
        form_type (str): The form type (default: 10-K)
        
    Returns:
        str: URL to the filing or None if not found
    """
    # Convert ticker to uppercase
    ticker = ticker.upper()
    
    # Search for the company on EDGAR
    search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to search for {ticker}. Status code: {response.status_code}")
            return None
        
        # Parse the search results page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with filings
        filing_table = soup.find('table', class_='tableFile2')
        if not filing_table:
            print(f"No filings found for {ticker}")
            return None
        
        # Look for the specified form type
        for row in filing_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                if cells[0].text.strip() == form_type:
                    # Found the form, get the documents link
                    doc_link = cells[1].find('a', href=True)
                    if doc_link and 'href' in doc_link.attrs:
                        documents_url = f"https://www.sec.gov{doc_link['href']}"
                        
                        # Get the documents page
                        doc_response = requests.get(documents_url, headers=headers)
                        if doc_response.status_code == 200:
                            doc_soup = BeautifulSoup(doc_response.text, 'html.parser')
                            
                            # Find the link to the HTML filing
                            for doc_row in doc_soup.find_all('tr'):
                                doc_cells = doc_row.find_all('td')
                                if len(doc_cells) >= 3:
                                    if '.htm' in doc_cells[2].text.lower() and not 'index' in doc_cells[2].text.lower():
                                        doc_url = doc_cells[2].find('a', href=True)
                                        if doc_url and 'href' in doc_url.attrs:
                                            return f"https://www.sec.gov{doc_url['href']}"
        
        print(f"Could not find {form_type} filing for {ticker}")
        return None
        
    except Exception as e:
        print(f"Error searching for {ticker}: {e}")
        traceback.print_exc()
        return None

def main():
    # Get ticker from command line or use default
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    print(f"Searching for {ticker} 10-K filings...")
    
    # Get the filing URL
    filing_url = get_sec_filing_url(ticker, "10-K")
    if not filing_url:
        print(f"Could not find 10-K filing for {ticker}")
        return 1
    
    print(f"Found filing: {filing_url}")
    
    # Download the filing
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        print("Downloading filing HTML content...")
        response = requests.get(filing_url, headers=headers)
        
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
            
            # Save the tables to Excel for inspection
            if tables:
                with pd.ExcelWriter(f"{ticker}_tables.xlsx") as writer:
                    # Save up to 30 tables
                    for i, table in enumerate(tables[:30]):
                        sheet_name = f"Table_{i}"
                        table.to_excel(writer, sheet_name=sheet_name)
                print(f"Saved tables to {ticker}_tables.xlsx")
            
            return 0
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
            traceback.print_exc()
            
            # Try to save a sample of the HTML for debugging
            with open(f"{ticker}_sample.html", "w", encoding="utf-8") as f:
                f.write(html_content[:10000])
            print(f"Saved a sample of the HTML to {ticker}_sample.html")
            
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
