"""
SEC EDGAR Tables - DEPRECATED

This module is deprecated. Please use sec_api_wrapper instead:

    from sec_api_wrapper import sec_api
    
    cik = sec_api.get_company_cik(ticker)
    filing_info = sec_api.get_latest_filing_info(cik, "10-K")
    html_content = sec_api.download_filing(filing_info)
    tables = sec_api.extract_tables(html_content)
    financial_tables = sec_api.identify_financial_tables(tables)

This module will be removed in a future version.
"""

import warnings
import requests
import pandas as pd
import sys
import os
import json
import time
from bs4 import BeautifulSoup
import traceback

# Emit deprecation warning when module is imported
warnings.warn(
    "sec_edgar_tables is deprecated. Use sec_api_wrapper instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

def get_company_cik(ticker):
    """Get the CIK number for a company ticker"""
    # SEC provides a JSON file with all CIK mappings
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get CIK data. Status code: {response.status_code}")
            return None
            
        data = response.json()
        
        # Find the ticker in the data
        ticker = ticker.upper()
        for _, company in data.items():
            if company["ticker"] == ticker:
                # Format CIK with leading zeros to 10 digits
                return str(company["cik_str"]).zfill(10)
                
        print(f"Could not find CIK for ticker {ticker}")
        return None
        
    except Exception as e:
        print(f"Error getting CIK: {e}")
        return None

def get_latest_filing(cik, form_type="10-K"):
    """Get the latest filing of specified type for a company"""
    # SEC EDGAR API endpoint for company submissions
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get filings. Status code: {response.status_code}")
            return None
            
        data = response.json()
        
        # Find the latest filing of the specified type
        recent_filings = data.get("filings", {}).get("recent", {})
        if not recent_filings:
            print("No recent filings found")
            return None
            
        form_types = recent_filings.get("form", [])
        accession_numbers = recent_filings.get("accessionNumber", [])
        filing_dates = recent_filings.get("filingDate", [])
        
        if not form_types or not accession_numbers:
            print("No form types or accession numbers found")
            return None
            
        # Find the latest filing of the specified type
        for i, form in enumerate(form_types):
            if form == form_type and i < len(accession_numbers):
                accession_number = accession_numbers[i].replace("-", "")
                filing_date = filing_dates[i] if i < len(filing_dates) else "unknown"
                
                # Construct the URL to the filing
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_number}/{accession_numbers[i]}-index.htm"
                
                return {
                    "url": filing_url,
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_date
                }
                
        print(f"No {form_type} filings found")
        return None
        
    except Exception as e:
        print(f"Error getting filing: {e}")
        traceback.print_exc()
        return None

def get_filing_document_url(index_url):
    """Get the URL to the actual filing document from the index page"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(index_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get index page. Status code: {response.status_code}")
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with the list of documents
        table = soup.find('table', summary='Document Format Files')
        if not table:
            print("Could not find document table")
            return None
            
        # Look for the 10-K document (usually has a description like "10-K" and is in HTML format)
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 3:
                description = cells[1].text.strip()
                document_link = cells[2].find('a')
                
                if document_link and ('10-k' in description.lower() or '.htm' in document_link.text.lower()):
                    document_url = f"https://www.sec.gov{document_link['href']}"
                    return document_url
                    
        # If we couldn't find a specific 10-K document, just get the first HTML document
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 3:
                document_link = cells[2].find('a')
                if document_link and '.htm' in document_link.text.lower():
                    document_url = f"https://www.sec.gov{document_link['href']}"
                    return document_url
                    
        print("Could not find filing document")
        return None
        
    except Exception as e:
        print(f"Error getting document URL: {e}")
        return None

def extract_tables_from_filing(document_url):
    """Extract tables from the filing document"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        print(f"Downloading filing from {document_url}")
        response = requests.get(document_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to get filing document. Status code: {response.status_code}")
            return None
            
        html_content = response.text
        print(f"Downloaded {len(html_content)} bytes of HTML content")
        
        # Save HTML content for debugging
        with open("filing.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("Saved HTML content to filing.html")
        
        # Extract tables from HTML
        print("Extracting tables from HTML...")
        tables = pd.read_html(html_content)
        print(f"Found {len(tables)} tables in the filing")
        
        return tables
        
    except Exception as e:
        print(f"Error extracting tables: {e}")
        traceback.print_exc()
        return None

def main():
    # Get ticker from command line or use default
    ticker = "AAPL"  # Default ticker
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    
    print(f"Processing {ticker} 10-K filing...")
    
    # Get the company CIK
    cik = get_company_cik(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return 1
    
    print(f"Found CIK: {cik}")
    
    # Get the latest 10-K filing
    filing_info = get_latest_filing(cik, "10-K")
    if not filing_info:
        print(f"Could not find 10-K filing for {ticker}")
        return 1
    
    print(f"Found filing: {filing_info['url']} dated {filing_info['filing_date']}")
    
    # Get the URL to the actual filing document
    document_url = get_filing_document_url(filing_info['url'])
    if not document_url:
        print("Could not find filing document URL")
        return 1
    
    print(f"Found document URL: {document_url}")
    
    # Extract tables from the filing
    tables = extract_tables_from_filing(document_url)
    if not tables:
        print("Could not extract tables from filing")
        return 1
    
    # Save the tables to Excel
    output_file = f"{ticker}_10K_tables.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        # Save up to 30 tables
        for i, table in enumerate(tables[:30]):
            sheet_name = f"Table_{i}"
            table.to_excel(writer, sheet_name=sheet_name)
    
    print(f"Saved {min(len(tables), 30)} tables to {output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
