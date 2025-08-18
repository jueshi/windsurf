#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for SEC filing extraction functionality.
This script tests the SEC filing extractor module with a sample ticker.
"""

import os
import sys
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import the SEC filing extractor module
import sec_filing_extractor

def test_sec_extraction(ticker, form_type):
    """Test SEC filing extraction for a given ticker and form type.
    
    Args:
        ticker (str): Stock ticker symbol
        form_type (str): Either '10-K' or '10-Q'
    
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    print(f"\n{'='*80}\nTesting SEC filing extraction for {ticker} ({form_type})\n{'='*80}")
    
    try:
        # Create output directory
        output_dir = os.path.join("sec_filings", ticker)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Step 1: Getting company CIK for {ticker}...")
        cik = sec_filing_extractor.get_company_cik(ticker)
        if not cik:
            print(f"Error: Could not find CIK for {ticker}")
            return False
        
        print(f"Found CIK: {cik}")
        
        print(f"\nStep 2: Getting latest {form_type} filing info...")
        filing_info = sec_filing_extractor.get_latest_filing_info(cik, form_type)
        if not filing_info:
            print(f"Error: Could not find {form_type} filing for {ticker}")
            return False
        
        print(f"Found {form_type} filing from {filing_info['filingDate']}")
        print(f"Filing URL: {filing_info['detailUrl']}")
        
        print("\nStep 3: Downloading filing...")
        html_content = sec_filing_extractor.download_filing(filing_info)
        if not html_content:
            print("Error: Failed to download filing")
            return False
        
        print(f"Successfully downloaded {len(html_content)} bytes")
        
        print("\nStep 4: Extracting tables...")
        tables = sec_filing_extractor.extract_tables(html_content)
        if not tables:
            print("Error: No tables found in filing")
            return False
        
        print(f"Found {len(tables)} tables")
        
        print("\nStep 5: Identifying financial tables...")
        financial_tables = sec_filing_extractor.identify_financial_tables(tables)
        
        # Count identified tables
        identified_count = sum(1 for table in financial_tables.values() if table is not None)
        print(f"Identified {identified_count} financial tables")
        
        print("\nStep 6: Saving tables to Excel...")
        success = sec_filing_extractor.save_tables_to_excel(financial_tables, tables, ticker, output_dir)
        
        if success:
            print(f"\nSuccessfully extracted and saved tables for {ticker}")
            print(f"\nFiles saved to: {os.path.abspath(output_dir)}")
            return True
        else:
            print("\nError: Failed to save tables to Excel")
            return False
            
    except Exception as e:
        print(f"\nError extracting {form_type} tables: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main function to run the test."""
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
    
    # Run the test
    start_time = datetime.now()
    success = test_sec_extraction(ticker, form_type)
    end_time = datetime.now()
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print(f"Execution time: {end_time - start_time}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
