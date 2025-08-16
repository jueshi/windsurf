#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the updated SEC filing extractor with rate limiting and error handling.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('test_extractor.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the SEC filing extractor module
import sec_filing_extractor

def main():
    """Main function to test the SEC filing extractor."""
    ticker = "AAPL"
    form_type = "10-K"
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    if len(sys.argv) > 2:
        form_type = sys.argv[2]
    
    output_dir = os.path.join("sec_filings", ticker)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Testing SEC Filing Extractor with {ticker} ({form_type})")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Get company CIK
        print("\nStep 1: Getting company CIK...")
        cik = sec_filing_extractor.get_company_cik(ticker)
        if not cik:
            print("Error: Could not get CIK. Test failed.")
            return
        
        # Step 2: Get latest filing info
        print("\nStep 2: Getting latest filing info...")
        filing_info = sec_filing_extractor.get_latest_filing_info(cik, form_type)
        if not filing_info:
            print("Error: Could not get filing info. Test failed.")
            return
        
        # Step 3: Download filing
        print("\nStep 3: Downloading filing...")
        html_content = sec_filing_extractor.download_filing(filing_info)
        if not html_content:
            print("Error: Could not download filing. Test failed.")
            return
        
        # Step 4: Extract tables
        print("\nStep 4: Extracting tables...")
        tables = sec_filing_extractor.extract_tables(html_content)
        if not tables:
            print("Error: Could not extract tables. Test failed.")
            return
        
        # Step 5: Identify financial tables
        print("\nStep 5: Identifying financial tables...")
        financial_tables = sec_filing_extractor.identify_financial_tables(tables)
        
        # Step 6: Save tables to Excel
        print("\nStep 6: Saving tables to Excel...")
        success = sec_filing_extractor.save_tables_to_excel(financial_tables, tables, ticker, output_dir)
        if not success:
            print("Error: Could not save tables to Excel. Test failed.")
            return
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"Test completed successfully in {duration}")
        print(f"Output saved to: {os.path.abspath(output_dir)}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        print("Test failed.")

if __name__ == "__main__":
    main()
