#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for SEC filing extractor module.
This script tests each function in the sec_filing_extractor module.
"""

import os
import sys
import time
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Log to stdout for immediate feedback
    ]
)

# Import the SEC filing extractor module
import sec_filing_extractor

def test_get_company_cik(ticker):
    """Test getting company CIK for a ticker."""
    print(f"\nTesting get_company_cik for {ticker}...")
    try:
        cik = sec_filing_extractor.get_company_cik(ticker)
        if cik:
            print(f"Success: Found CIK {cik} for {ticker}")
            return cik
        else:
            print(f"Error: Could not find CIK for {ticker}")
            return None
    except Exception as e:
        print(f"Exception in get_company_cik: {str(e)}")
        traceback.print_exc()
        return None

def test_get_latest_filing_info(cik, form_type):
    """Test getting latest filing info for a CIK and form type."""
    print(f"\nTesting get_latest_filing_info for CIK {cik}, form type {form_type}...")
    try:
        filing_info = sec_filing_extractor.get_latest_filing_info(cik, form_type)
        if filing_info:
            print(f"Success: Found {form_type} filing from {filing_info.get('filingDate', 'unknown date')}")
            print(f"Filing URL: {filing_info.get('detailUrl', 'unknown URL')}")
            return filing_info
        else:
            print(f"Error: Could not find {form_type} filing for CIK {cik}")
            return None
    except Exception as e:
        print(f"Exception in get_latest_filing_info: {str(e)}")
        traceback.print_exc()
        return None

def test_download_filing(filing_info):
    """Test downloading a filing."""
    print("\nTesting download_filing...")
    try:
        if not filing_info or 'detailUrl' not in filing_info:
            print("Error: Invalid filing info")
            return None
            
        html_content = sec_filing_extractor.download_filing(filing_info)
        if html_content:
            print(f"Success: Downloaded {len(html_content)} bytes")
            return html_content
        else:
            print("Error: Failed to download filing")
            return None
    except Exception as e:
        print(f"Exception in download_filing: {str(e)}")
        traceback.print_exc()
        return None

def test_extract_tables(html_content):
    """Test extracting tables from HTML content."""
    print("\nTesting extract_tables...")
    try:
        if not html_content:
            print("Error: No HTML content provided")
            return None
            
        tables = sec_filing_extractor.extract_tables(html_content)
        if tables:
            print(f"Success: Extracted {len(tables)} tables")
            return tables
        else:
            print("Error: No tables found in filing")
            return None
    except Exception as e:
        print(f"Exception in extract_tables: {str(e)}")
        traceback.print_exc()
        return None

def test_identify_financial_tables(tables):
    """Test identifying financial tables."""
    print("\nTesting identify_financial_tables...")
    try:
        if not tables:
            print("Error: No tables provided")
            return None
            
        financial_tables = sec_filing_extractor.identify_financial_tables(tables)
        if financial_tables:
            identified_count = sum(1 for table in financial_tables.values() if table is not None)
            print(f"Success: Identified {identified_count} financial tables")
            return financial_tables
        else:
            print("Error: Failed to identify financial tables")
            return None
    except Exception as e:
        print(f"Exception in identify_financial_tables: {str(e)}")
        traceback.print_exc()
        return None

def test_save_tables_to_excel(financial_tables, tables, ticker, output_dir):
    """Test saving tables to Excel."""
    print("\nTesting save_tables_to_excel...")
    try:
        if not financial_tables or not tables:
            print("Error: No tables provided")
            return False
            
        success = sec_filing_extractor.save_tables_to_excel(financial_tables, tables, ticker, output_dir)
        if success:
            print(f"Success: Saved tables to {output_dir}")
            return True
        else:
            print(f"Error: Failed to save tables to {output_dir}")
            return False
    except Exception as e:
        print(f"Exception in save_tables_to_excel: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function to run the tests."""
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
    
    print(f"\n{'='*80}\nTesting SEC filing extractor with {ticker} ({form_type})\n{'='*80}")
    
    # Create output directory
    output_dir = os.path.join("sec_filings", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the tests
    start_time = datetime.now()
    
    # Test each function in sequence
    cik = test_get_company_cik(ticker)
    if not cik:
        print("\nTest failed at step 1: Could not get CIK")
        return
        
    filing_info = test_get_latest_filing_info(cik, form_type)
    if not filing_info:
        print("\nTest failed at step 2: Could not get filing info")
        return
        
    html_content = test_download_filing(filing_info)
    if not html_content:
        print("\nTest failed at step 3: Could not download filing")
        return
        
    tables = test_extract_tables(html_content)
    if not tables:
        print("\nTest failed at step 4: Could not extract tables")
        return
        
    financial_tables = test_identify_financial_tables(tables)
    if not financial_tables:
        print("\nTest failed at step 5: Could not identify financial tables")
        return
        
    success = test_save_tables_to_excel(financial_tables, tables, ticker, output_dir)
    if not success:
        print("\nTest failed at step 6: Could not save tables to Excel")
        return
    
    end_time = datetime.now()
    
    # Print results
    print(f"\n{'='*80}")
    print(f"All tests PASSED!")
    print(f"Execution time: {end_time - start_time}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
