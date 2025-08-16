#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SEC API Wrapper - Provides a unified interface for both real SEC API and mock SEC data
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Import mock SEC data provider
from mock_sec_data import MockSECAPI

# Import SEC API cache module
import sec_api_cache

# Load environment variables
load_dotenv()

class SECAPIWrapper:
    """
    Wrapper for SEC API that can switch between real and mock data
    """
    
    def __init__(self, use_mock=False):
        """
        Initialize SEC API wrapper
        
        Args:
            use_mock (bool): Whether to use mock data instead of real SEC API
        """
        self.use_mock = use_mock
        
        # Initialize mock API if needed
        if self.use_mock:
            self.mock_api = MockSECAPI()
            print("Using mock SEC data for testing")
        else:
            print("Using real SEC API (with caching and rate limiting)")
    
    def get_company_cik(self, ticker):
        """
        Get company CIK from ticker
        
        Args:
            ticker (str): Company ticker symbol
            
        Returns:
            str: CIK number (10 digits with leading zeros) or None if not found
        """
        if self.use_mock:
            return self.mock_api.get_company_cik(ticker)
        else:
            return sec_api_cache.get_company_cik(ticker)
    
    def get_latest_filing_info(self, cik, form_type="10-K"):
        """
        Get latest filing info for a company
        
        Args:
            cik (str): Company CIK number (10 digits with leading zeros)
            form_type (str): Form type to search for (10-K, 10-Q, etc.)
            
        Returns:
            dict: Filing information or None if not found
        """
        if self.use_mock:
            return self.mock_api.get_latest_filing_info(cik, form_type)
        else:
            return sec_api_cache.get_latest_filing_info(cik, form_type)
    
    def download_filing(self, filing_info):
        """
        Download filing document
        
        Args:
            filing_info (dict): Filing information from get_latest_filing_info
            
        Returns:
            str: HTML content of filing or None if download fails
        """
        if self.use_mock:
            return self.mock_api.download_filing(filing_info)
        else:
            return sec_api_cache.download_filing(filing_info)

# Create a global instance with default settings
# This can be imported and used directly by other modules
sec_api = SECAPIWrapper(use_mock=False)

# Function to switch between real and mock API
def use_mock_sec_api(use_mock=True):
    """
    Switch between real and mock SEC API
    
    Args:
        use_mock (bool): Whether to use mock data
    
    Returns:
        SECAPIWrapper: New SEC API wrapper instance
    """
    global sec_api
    sec_api = SECAPIWrapper(use_mock=use_mock)
    return sec_api

# Function to check if we're using mock API
def using_mock_api():
    """
    Check if we're currently using the mock SEC API
    
    Returns:
        bool: True if using mock API, False if using real API
    """
    global sec_api
    return sec_api.use_mock

# Test function
def test_sec_api_wrapper(use_mock=True, ticker="AAPL"):
    """
    Test SEC API wrapper
    
    Args:
        use_mock (bool): Whether to use mock data
        ticker (str): Ticker to test with
    """
    # Switch to mock or real API
    api = use_mock_sec_api(use_mock)
    
    print(f"\n{'='*80}")
    print(f"Testing SEC API Wrapper with {'MOCK' if use_mock else 'REAL'} data")
    print(f"Ticker: {ticker}")
    print(f"{'='*80}\n")
    
    # Step 1: Get company CIK
    print("\nStep 1: Getting company CIK...")
    cik = api.get_company_cik(ticker)
    print(f"CIK result: {cik}")
    
    if not cik:
        print("Error: Could not get CIK. Test failed.")
        return
    
    # Step 2: Get latest filing info for 10-K
    print("\nStep 2: Getting latest 10-K filing info...")
    filing_info_10k = api.get_latest_filing_info(cik, "10-K")
    print(f"10-K Filing info: {filing_info_10k}")
    
    if filing_info_10k:
        # Step 3: Download 10-K filing
        print("\nStep 3: Downloading 10-K filing...")
        html_content = api.download_filing(filing_info_10k)
        
        if html_content:
            print(f"Downloaded 10-K HTML content: {len(html_content)} bytes")
        else:
            print("Error: Could not download 10-K filing.")
    else:
        print("Error: Could not get 10-K filing info.")
    
    # Step 4: Get latest filing info for 10-Q
    print("\nStep 4: Getting latest 10-Q filing info...")
    filing_info_10q = api.get_latest_filing_info(cik, "10-Q")
    print(f"10-Q Filing info: {filing_info_10q}")
    
    if filing_info_10q:
        # Step 5: Download 10-Q filing
        print("\nStep 5: Downloading 10-Q filing...")
        html_content = api.download_filing(filing_info_10q)
        
        if html_content:
            print(f"Downloaded 10-Q HTML content: {len(html_content)} bytes")
        else:
            print("Error: Could not download 10-Q filing.")
    else:
        print("Error: Could not get 10-Q filing info.")
    
    print("\nTest completed")

if __name__ == "__main__":
    import sys
    
    # Default to mock data
    use_mock = True
    ticker = "AAPL"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "real":
            use_mock = False
        elif sys.argv[1].lower() == "mock":
            use_mock = True
        else:
            ticker = sys.argv[1].upper()
    
    if len(sys.argv) > 2 and sys.argv[1].lower() not in ["real", "mock"]:
        ticker = sys.argv[2].upper()
    
    # Run test
    test_sec_api_wrapper(use_mock, ticker)
