"""
Simple test script for the SEC API wrapper with mock data.
This script tests the basic functionality of the mock SEC data provider.
"""

import os
import sys
import logging
import sec_api_wrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_mock_sec_api.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def test_mock_sec_api():
    """Test the SEC API wrapper with mock data."""
    print("Testing SEC API wrapper with mock data")
    
    # Enable mock SEC API
    sec_api_wrapper.use_mock_sec_api(True)
    print(f"Using mock SEC API: {sec_api_wrapper.using_mock_api()}")
    
    # Get the API instance
    api = sec_api_wrapper.sec_api
    
    # Test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"\nTesting with ticker: {ticker}")
        
        # Get company CIK
        print("Getting company CIK...")
        cik = api.get_company_cik(ticker)
        print(f"CIK for {ticker}: {cik}")
        
        if not cik:
            print(f"Error: Could not find CIK for {ticker}")
            continue
        
        # Get latest 10-K filing info
        print("Getting latest 10-K filing info...")
        filing_info = api.get_latest_filing_info(cik, "10-K")
        print(f"10-K filing info for {ticker}: {filing_info['filingDate'] if filing_info else 'Not found'}")
        
        if not filing_info:
            print(f"Error: Could not find 10-K filing for {ticker}")
            continue
        
        # Download filing
        print("Downloading filing...")
        html_content = api.download_filing(filing_info)
        print(f"Downloaded {len(html_content) if html_content else 0} bytes")
        
        if not html_content:
            print(f"Error: Failed to download filing for {ticker}")
            continue
        
        print(f"Successfully tested mock SEC API with {ticker}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_mock_sec_api()
