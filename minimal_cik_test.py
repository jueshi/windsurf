#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal test for SEC API connectivity - just testing CIK lookup
"""

import os
import sys
import requests
import time
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get SEC email
email = os.getenv("SEC_EDGAR_EMAIL", "jueshi@gmail.com")

def get_headers():
    """Get headers for SEC API requests"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "From": email
    }
    return headers

def make_request(url, max_retries=3):
    """Make request with retries and backoff"""
    headers = get_headers()
    
    # Open output file
    with open("minimal_cik_output.txt", "a", encoding="utf-8") as f:
        f.write(f"Requesting URL: {url}\n")
        f.write(f"Headers: {headers}\n")
        f.flush()
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    f.write(f"Retry {attempt+1}/{max_retries}. Waiting {delay:.2f} seconds...\n")
                    f.flush()
                    time.sleep(delay)
                
                f.write(f"Sending request...\n")
                f.flush()
                
                response = requests.get(url, headers=headers, timeout=30)
                
                f.write(f"Response status code: {response.status_code}\n")
                f.write(f"Response headers: {response.headers}\n")
                f.flush()
                
                if response.status_code == 200:
                    return response
                else:
                    f.write(f"Error response content: {response.text[:500]}...\n")
                    f.flush()
            
            except Exception as e:
                f.write(f"Request error: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
        
        f.write(f"Failed after {max_retries} attempts\n")
        f.flush()
        return None

def main():
    """Main test function"""
    # Clear previous output
    with open("minimal_cik_output.txt", "w", encoding="utf-8") as f:
        f.write(f"=== Minimal CIK Test ===\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Email: {email}\n\n")
    
    ticker = "AAPL"
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    
    # Test company tickers endpoint
    url = "https://www.sec.gov/files/company_tickers.json"
    
    with open("minimal_cik_output.txt", "a", encoding="utf-8") as f:
        f.write(f"Looking up CIK for {ticker}...\n")
        f.flush()
    
    response = make_request(url)
    
    if response and response.status_code == 200:
        with open("minimal_cik_output.txt", "a", encoding="utf-8") as f:
            try:
                data = response.json()
                f.write(f"Successfully retrieved data with {len(data)} entries\n")
                
                # Find the company by ticker
                found = False
                for _, company in data.items():
                    if company.get("ticker", "").upper() == ticker.upper():
                        cik = str(company.get("cik_str", "")).zfill(10)
                        f.write(f"Found CIK: {cik} for {company.get('title', '')}\n")
                        found = True
                        break
                
                if not found:
                    f.write(f"Could not find ticker {ticker} in the data\n")
                
                f.write("\nTest completed successfully\n")
                f.flush()
                
            except Exception as e:
                f.write(f"Error processing response: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
                f.flush()
    else:
        with open("minimal_cik_output.txt", "a", encoding="utf-8") as f:
            f.write("Test failed - could not retrieve company tickers\n")
            f.flush()

if __name__ == "__main__":
    main()
