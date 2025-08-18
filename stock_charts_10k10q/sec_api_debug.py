#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script for SEC API connectivity.
Tests basic HTTP requests to SEC.gov endpoints with detailed error reporting.
"""

import os
import sys
import requests
import json
import time
import logging
from datetime import datetime

# Configure logging to file and stdout
log_file = 'sec_api_debug.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure output to be unbuffered
sys.stdout.reconfigure(line_buffering=True)

def print_separator(title):
    """Print a separator with title."""
    separator = f"\n{'='*80}\n  {title}\n{'='*80}\n"
    print(separator)
    logging.info(separator)

def get_headers():
    """Get headers for SEC EDGAR API requests."""
    logging.info("Setting up request headers...")
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

def test_company_tickers_endpoint():
    """Test the SEC company tickers endpoint."""
    print_separator("Testing SEC Company Tickers Endpoint")
    
    url = "https://www.sec.gov/files/company_tickers.json"
    print(f"Requesting URL: {url}")
    
    headers = get_headers()
    print(f"Using headers: {headers}")
    
    try:
        print("Sending request...")
        start_time = datetime.now()
        response = requests.get(url, headers=headers, timeout=30)
        end_time = datetime.now()
        
        print(f"Request completed in {end_time - start_time}")
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully retrieved data with {len(data)} entries")
            
            # Try to find Apple as a test
            for _, company in data.items():
                if company.get('ticker') == 'AAPL':
                    print(f"Found AAPL: CIK = {company.get('cik_str')}")
                    break
            return True
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_company_submissions_endpoint(cik):
    """Test the SEC company submissions endpoint."""
    print_separator(f"Testing SEC Company Submissions Endpoint for CIK {cik}")
    
    # Format CIK with leading zeros to 10 digits
    cik = str(cik).zfill(10)
    
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    print(f"Requesting URL: {url}")
    
    headers = get_headers()
    print(f"Using headers: {headers}")
    
    try:
        print("Sending request...")
        start_time = datetime.now()
        response = requests.get(url, headers=headers, timeout=30)
        end_time = datetime.now()
        
        print(f"Request completed in {end_time - start_time}")
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Successfully retrieved data")
            
            # Print company name
            if 'name' in data:
                print(f"Company name: {data['name']}")
                
            # Check for recent filings
            recent_filings = data.get("filings", {}).get("recent", {})
            if recent_filings:
                form_types = recent_filings.get("form", [])
                filing_dates = recent_filings.get("filingDate", [])
                
                print(f"Found {len(form_types)} recent filings")
                
                # Print the first 5 filings
                for i in range(min(5, len(form_types))):
                    print(f"  {filing_dates[i]}: {form_types[i]}")
                    
                # Look for 10-K filings
                ten_k_indices = [i for i, form in enumerate(form_types) if form == '10-K']
                if ten_k_indices:
                    print(f"Found {len(ten_k_indices)} 10-K filings")
                    for i in ten_k_indices[:3]:  # Show up to 3
                        print(f"  10-K filed on {filing_dates[i]}")
                else:
                    print("No 10-K filings found")
                    
                return True
            else:
                print("No recent filings found")
                return False
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_filing_document_endpoint():
    """Test accessing a filing document."""
    print_separator("Testing SEC Filing Document Endpoint")
    
    # Apple's 10-K from 2022 as an example
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm"
    print(f"Requesting URL: {url}")
    
    headers = get_headers()
    print(f"Using headers: {headers}")
    
    try:
        print("Sending request...")
        start_time = datetime.now()
        response = requests.get(url, headers=headers, timeout=30)
        end_time = datetime.now()
        
        print(f"Request completed in {end_time - start_time}")
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            html_content = response.text
            print(f"Successfully retrieved HTML content ({len(html_content)} bytes)")
            
            # Check for table tags
            table_count = html_content.count("<table")
            print(f"Found {table_count} table tags in the HTML")
            
            # Print a small sample of the HTML
            print("HTML sample (first 500 chars):")
            print(html_content[:500])
            
            return True
        else:
            print(f"Failed with status code: {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the tests."""
    print_separator("SEC API Debug Tests")
    print(f"Current time: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Requests version: {requests.__version__}")
    
    # Test company tickers endpoint
    if not test_company_tickers_endpoint():
        print("Company tickers endpoint test failed")
        return
    
    # Wait to avoid rate limiting
    print("\nWaiting 2 seconds to avoid rate limiting...")
    time.sleep(2)
    
    # Test company submissions endpoint for Apple (CIK: 320193)
    if not test_company_submissions_endpoint(320193):
        print("Company submissions endpoint test failed")
        return
    
    # Wait to avoid rate limiting
    print("\nWaiting 2 seconds to avoid rate limiting...")
    time.sleep(2)
    
    # Test filing document endpoint
    if not test_filing_document_endpoint():
        print("Filing document endpoint test failed")
        return
    
    print_separator("All Tests Completed Successfully")

if __name__ == "__main__":
    main()
