#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script for SEC filing extractor module.
This script tests each function in the sec_filing_extractor module with detailed debug output.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime

# Configure logging to stdout with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout for immediate feedback
    ]
)

# Import the SEC filing extractor module
import sec_filing_extractor

def debug_get_company_cik(ticker):
    """Test getting company CIK for a ticker with detailed debug output."""
    print(f"\n{'*'*80}\nDEBUG: Testing get_company_cik for {ticker}...\n{'*'*80}")
    try:
        # Get the headers first to verify they're correct
        headers = sec_filing_extractor.get_headers()
        print(f"DEBUG: Using headers: {headers}")
        
        # Make the request manually to verify response
        url = "https://www.sec.gov/files/company_tickers.json"
        print(f"DEBUG: Sending request to {url}")
        
        import requests
        response = requests.get(url, headers=headers)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response headers: {response.headers}")
        
        if response.status_code != 200:
            print(f"DEBUG: Error response: {response.text[:500]}...")
            return None
            
        # Parse a small sample of the response to verify it's valid JSON
        sample_data = response.json()
        print(f"DEBUG: Sample data (first 3 entries):")
        for i in range(3):
            if str(i) in sample_data:
                print(f"  {i}: {sample_data[str(i)]}")
        
        # Now call the actual function
        print(f"DEBUG: Calling get_company_cik({ticker})...")
        cik = sec_filing_extractor.get_company_cik(ticker)
        print(f"DEBUG: get_company_cik returned: {cik}")
        return cik
        
    except Exception as e:
        print(f"DEBUG: Exception in debug_get_company_cik: {str(e)}")
        traceback.print_exc()
        return None

def debug_get_latest_filing_info(cik, form_type):
    """Test getting latest filing info with detailed debug output."""
    print(f"\n{'*'*80}\nDEBUG: Testing get_latest_filing_info for CIK {cik}, form type {form_type}...\n{'*'*80}")
    try:
        # Get the headers
        headers = sec_filing_extractor.get_headers()
        
        # Make the request manually
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"DEBUG: Sending request to {url}")
        
        import requests
        response = requests.get(url, headers=headers)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response headers: {response.headers}")
        
        if response.status_code != 200:
            print(f"DEBUG: Error response: {response.text[:500]}...")
            return None
            
        # Now call the actual function
        print(f"DEBUG: Calling get_latest_filing_info({cik}, {form_type})...")
        filing_info = sec_filing_extractor.get_latest_filing_info(cik, form_type)
        print(f"DEBUG: get_latest_filing_info returned: {filing_info}")
        return filing_info
        
    except Exception as e:
        print(f"DEBUG: Exception in debug_get_latest_filing_info: {str(e)}")
        traceback.print_exc()
        return None

def debug_download_filing(filing_info):
    """Test downloading a filing with detailed debug output."""
    print(f"\n{'*'*80}\nDEBUG: Testing download_filing...\n{'*'*80}")
    try:
        if not filing_info or 'detailUrl' not in filing_info:
            print("DEBUG: Invalid filing info")
            return None
            
        # Get the headers
        headers = sec_filing_extractor.get_headers()
        
        # Make the request manually
        url = filing_info['detailUrl']
        print(f"DEBUG: Sending request to {url}")
        
        import requests
        response = requests.get(url, headers=headers)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response headers: {response.headers}")
        
        if response.status_code != 200:
            print(f"DEBUG: Error response: {response.text[:500]}...")
            return None
            
        # Print a small sample of the response
        print(f"DEBUG: Response content sample (first 500 chars):")
        print(response.text[:500])
        
        # Now call the actual function
        print(f"DEBUG: Calling download_filing...")
        html_content = sec_filing_extractor.download_filing(filing_info)
        print(f"DEBUG: download_filing returned {len(html_content) if html_content else 'None'} bytes")
        return html_content
        
    except Exception as e:
        print(f"DEBUG: Exception in debug_download_filing: {str(e)}")
        traceback.print_exc()
        return None

def debug_extract_tables(html_content):
    """Test extracting tables with detailed debug output."""
    print(f"\n{'*'*80}\nDEBUG: Testing extract_tables...\n{'*'*80}")
    try:
        if not html_content:
            print("DEBUG: No HTML content provided")
            return None
            
        # Check if the HTML content contains table tags
        table_count = html_content.count("<table")
        print(f"DEBUG: HTML content contains {table_count} table tags")
        
        # Now call the actual function
        print(f"DEBUG: Calling extract_tables on {len(html_content)} bytes of HTML...")
        tables = sec_filing_extractor.extract_tables(html_content)
        print(f"DEBUG: extract_tables returned {len(tables) if tables else 0} tables")
        
        # Print info about the first few tables
        if tables:
            print(f"DEBUG: First 3 tables info:")
            for i, table in enumerate(tables[:3]):
                print(f"  Table {i}: Shape {table.shape}, Columns: {list(table.columns)}")
                print(f"  Sample data:\n{table.head(2)}")
                
        return tables
        
    except Exception as e:
        print(f"DEBUG: Exception in debug_extract_tables: {str(e)}")
        traceback.print_exc()
        return None

def main():
    """Main function to run the debug tests."""
    # Default test ticker
    ticker = "AAPL"
    form_type = "10-K"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        form_type = sys.argv[2]
        
    # Validate form type
    if form_type not in ["10-K", "10-Q"]:
        print("Error: Form type must be either '10-K' or '10-Q'")
        return
    
    print(f"\n{'='*80}\nDEBUG: Testing SEC filing extractor with {ticker} ({form_type})\n{'='*80}")
    
    # Create output directory
    output_dir = os.path.join("sec_filings", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the debug tests
    start_time = datetime.now()
    
    # Test each function in sequence with detailed debug output
    cik = debug_get_company_cik(ticker)
    if not cik:
        print("\nDEBUG: Test failed at step 1: Could not get CIK")
        return
        
    filing_info = debug_get_latest_filing_info(cik, form_type)
    if not filing_info:
        print("\nDEBUG: Test failed at step 2: Could not get filing info")
        return
        
    html_content = debug_download_filing(filing_info)
    if not html_content:
        print("\nDEBUG: Test failed at step 3: Could not download filing")
        return
        
    tables = debug_extract_tables(html_content)
    if not tables:
        print("\nDEBUG: Test failed at step 4: Could not extract tables")
        return
    
    end_time = datetime.now()
    
    # Print results
    print(f"\n{'='*80}")
    print(f"DEBUG: Test completed in {end_time - start_time}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
